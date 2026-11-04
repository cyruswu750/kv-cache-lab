# M0 代码导读（中文）

## 1. 执行流程（`scripts/m0_experiment.py` → `kvlab/experiment.py: run_case`）

对一个长度为 N 的 prompt、chunk 大小 C=256：

1. **选边界**：`eligible_boundaries(N, C)` 给出所有可恢复长度 L（`L % C == 0` 且 `0 < L ≤ N−1`）。为空 → 只跑路径 A，记录"无可恢复 chunk"。
2. **路径 A（完整重算）**：`prefill(ids[0:N], cache=None, start_pos=0)` 在一个**新的** `DynamicCache` 上算出首 token logits，再用 `greedy_continue` 贪心解码 8 个 token。随后丢弃这个 cache。
3. **路径 B 前半（导出）**：在**另一个新 cache** 上只算前缀 `ids[0:Lmax]`，然后 `export_prefix_to_cpu` 按 chunk、按层把 K/V 拷到新分配的 CPU 张量。
   - 立即 `compare_snapshot_to_cache` 逐张量做字节比较（导出检查）。
   - 记录每个张量的 SHA-256。
   - **删除 GPU 前缀 cache** 并 `empty_cache()`，此后 GPU 上不再有原始前缀数据。
4. **路径 B 后半（恢复 + 续算）**：对每个合法 L（含部分恢复，如 N=513 时的 L=256）：
   - `restore_to_device(snapshot, L)` 用前 L/C 个 chunk 构造一个全新的 `DynamicCache`。
   - 再做一次字节比较（恢复检查：GPU 上的恢复结果 vs CPU 快照）。
   - `prefill(ids[L:N], cache, start_pos=L)` 只喂入后缀 token，然后同样贪心解码 8 个。
   - 与路径 A 比较：首 token logits（max_abs、rel_l2、allclose、argmax）+ 贪心序列。
   - 信息性检查：与"GPU 常驻的分块 prefill"（不经过 CPU）逐位比较 logits。
   - 负对照（仅 L=Lmax）：恢复一个全零快照，logits 必须明显偏离 A。
5. 最后核对快照 SHA-256 没变；按 `case_passed` 判定 PASS/FAIL。

## 2. 关键函数

| 函数 | 作用 | 要点 |
|---|---|---|
| `boundaries.eligible_boundaries` | 首 token 规则 | N=256 → `[]`；N=512 → `[256]`（最后一个完整 chunk 不恢复） |
| `kv_snapshot.export_prefix_to_cpu` | GPU cache → CPU 快照 | `torch.empty(..., device="cpu")` + `copy_`，保证独立存储；切片非连续也能正确拷贝 |
| `kv_snapshot.restore_to_device` | CPU 快照 → 新 GPU cache | 先分配设备张量、逐 chunk H2D 拷入，再用公开构造函数 `DynamicCache(ddp_cache_data=..., config=...)` |
| `kv_snapshot.compare_snapshot_to_cache` | 严格字节比较 | 转成 `uint8` 视图后 `torch.equal`；+0.0/−0.0 会被区分 |
| `model.prefill` | 显式位置的前向 | 显式传 `position_ids = arange(L, L+n)`、2D 全 1 `attention_mask`（长度 L+n）；前后断言 cache 长度 = L 和 L+n |
| `model.greedy_continue` | 贪心解码 | 每步喂 1 个 token，位置递增 |
| `metrics.compare_logits` | 数值比较 | float32 下算 max_abs、rel_l2、elementwise allclose、top-2 margin |
| `experiment.case_passed` | 判定 | 规则在比较之前就固定了（decisions.md D7） |

## 3. Cache 所有权（谁拥有哪块内存）

- **路径 A 的 cache**：模型前向时新建，只属于路径 A，用完即删。
- **前缀 cache（GPU）**：只用于导出。导出和字节检查做完就删掉，**恢复前它已不存在**。
- **CPU 快照**：`chunks[c][layer] = (K, V)`，每个张量都是新分配、连续、独立的 CPU 存储，不与任何 GPU 张量或其他快照张量共享 storage（CPU 测试检查了 `data_ptr` 不相交）。整个过程只读，用 SHA-256 前后核对。
- **恢复出的 cache（GPU）**：每次恢复都新建。`DynamicLayer.update` 内部做 `torch.cat`，所以 cache 拥有自己的拷贝。之后续算和解码会往里追加（cache 是可变对象），但不会碰到快照；两次恢复得到的 cache 彼此独立（CPU 测试覆盖）。
- **不在测试之间共享 cache**：每个 `run_case` 和每个 L 都重新构造 cache；GPU 测试只在模块级共享（只读的）模型权重。

## 4. 位置处理

- transformers 5.22.1 的 Qwen2 **没有** `cache_position` 参数；不传 `position_ids` 时，它用 `arange(n) + cache.get_seq_length()`。
- 我们显式传 `position_ids = arange(L, N)`，并断言前向前 `cache.get_seq_length() == L`。这样"恢复的 token 数""RoPE 位置""mask 长度"三者不可能悄悄不一致。
- 注意：K 在写入 cache **之前**已经施加了 RoPE（按绝对位置）。所以恢复的前缀必须放在原来的位置 `[0, L)`，不能平移。
- mask：2D `attention_mask` 的长度是 `L + n`（过去 + 当前），由 `create_causal_mask` 结合 cache 长度生成因果 mask。
- 解码第 i 步：位置为 `N + i`，mask 长度为 `N + i + 1`。

## 5. 边界情况

- N=255、256：没有合法 L，只跑路径 A。N=256 恰好是 chunk 整数倍，但唯一的 chunk 包含最后一个 prompt token，不能恢复。
- N=257：L=256，后缀只有 1 个 token，是"恢复量最大"的情况。
- N=513：L ∈ {256, 512}；L=256 是部分恢复（快照有 2 个 chunk，只用第 1 个）。
- 数值：分块 prefill 与完整 prefill 在 bf16 下本来就有差异（校准测得最多 0.35），这与存储无关；恢复路径与 GPU 常驻的分块路径**逐位相同**。
- 失败案例 `nat_long`：参考路径在第 6 步出现 bf16 **完全平局**（两个 token 都是 23.75），分块计算让其中一个多了 1 ulp，贪心序列从此分叉。按预先固定的规则仍判 FAIL，未放宽。

## 6. 你应该能回答的五个问题

1. 为什么 N=512、C=256 时只恢复 256 个 token，而不是 512？如果恢复 512，第一个输出 token 的 logits 从哪里来？
2. 恢复后的 cache 为什么一定来自 CPU 快照，而不是残留的 GPU 前缀 cache 或一个别名？请举出代码中的三条证据。
3. 续算时 `position_ids` 应该从几开始？如果错误地从 0 开始，K 已经带有的 RoPE 位置和新 Q 的位置会出现什么不一致？为什么字节比较发现不了这个错误？
4. 字节完全相等，为什么首 token logits 还会和完整重算差 0.2–0.56？哪一项检查说明这个差异来自"分块计算"而不是"存储往返"？
5. `nat_long` 失败时，哪些信息说明它是数值平局而不是 bug？如果要为 M1 改判定规则，怎样做才不算"为通过而放宽"？
