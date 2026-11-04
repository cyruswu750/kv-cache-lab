# M1 代码导读（中文）

M1 把 M0 的"一次性 CPU 快照"换成了真正的两层前缀缓存：pinned host + 磁盘。
策略仍是同步的"总是恢复"。设计约定见 `docs/design.md`，取舍见 `docs/decisions.md` D11–D21。

> **M1.1 修改了哪些行为**（详见 `docs/M1_1_REVIEW.md`）：
> 1. 磁盘文件**先删除成功，再归还预算**。删除失败的文件记为 undeletable：仍计入预算，永不再读，也不会自动重试。
> 2. 打开已有目录时，先完整校验（不跟随符号链接），再删除或写入任何东西。marker 必须是普通文件，已有的 marker 绝不改写。
> 3. host 计数拆开报告：slab 分配量、CUDA 注册量、staging 预留量、逻辑占用量（含峰值）。
> 4. 推理行为没有改变。D10 已决定：不加平局规则，M0 strict 的 FAIL 按原样保留。

## 1. 一次请求的执行流程（`kvlab/always_restore.py: AlwaysRestoreEngine.run`）

1. **算 key**：`chain_keys(ns_digest, token_ids, C)` 为 prompt 的每个**完整** chunk 算出链式 key。
2. **定上限**：`max_restore_len(N, C)` 给出允许恢复的最大 L（L % C == 0 且 L ≤ N−1），计划块数 `n_plan = L/C`。
3. **恢复**：`TieredKVCache.restore` 从块 0 开始沿链查找。
   - 先只看索引（host 或 disk 里有没有），得到连续存在的块数 n。
   - 然后逐块读取并校验：host 块在 lease 下直接 H2D；disk 块先读进 staging、完成校验，再 H2D。
   - 第一个失败的块处停止，只用之前已验证的连续前缀 L。
   - 最后用 `[0, L)` 构造**全新的** `DynamicCache`。
4. **续算**：`prefill(ids[L:], cache, start_pos=L)`；L = 0 时完整 prefill。位置和 mask 的规则与 M0 相同（显式 `position_ids`，并守卫 cache 长度）。
5. **写入**：`TieredKVCache.store` 把还没缓存的完整 prompt chunk 用 D2H 直接写进 host slot。host 满时先按 LRU 把最旧且未 pin 的块同步溢写到磁盘。
6. **解码**：贪心解码 8 个 token，然后删掉 GPU cache。下一个请求只能通过分层存储复用 KV。

## 2. 关键模块与函数

| 模块 / 函数 | 作用 | 要点 |
|---|---|---|
| `identity.Namespace` / `namespace_for` | 缓存命名空间 | 字段包括模型 id/revision、实际 KV dtype、层/头/维、chunk 大小、RoPE、layer_types、注意力实现、transformers/torch 版本、布局版本；规范 JSON + SHA-256 |
| `identity.chunk_key` / `chain_keys` | 链式 key | `SHA-256(tag + ns_digest + parent + u32 index + u32 count + u32 tokens)`；首块 parent 为 32 个 0 |
| `chunk_format.encode_header` / `write_chunk` | 写文件 | 140 B 定长头 + header CRC + tokens + meta + payload + payload CRC，全部小端 |
| `chunk_format.read_chunk_into` | 校验并读取 | 先做所有有界检查，再读 payload；失败抛 `ChunkReadError(reason)` |
| `host_store.HostStore` | host 层 | 一块 `cudaHostRegister` 的固定 slab；`allocate(spill)` 负责 LRU 淘汰和溢写；`lease` 负责 pin；`staging()` 是唯一的暂存缓冲；计数分为 `slab_bytes` / `registered_bytes` / `staging_bytes` / `resident_bytes`（逻辑占用）/ `peak_resident_bytes` |
| `disk_store.DiskStore` | 磁盘层 | 带 marker 的专属目录（先校验再动手）、内部生成的文件名、tmp → `os.replace`；tmp 字节先预留进预算；`finally` 中清理 tmp；读到坏文件即删除。`_unlink` 只处理 `OSError`，删除失败记入 `undeletable` |
| `tiered_cache.TieredKVCache` | 组合两层 | `store` / `restore` / `_spill` / 计数器 / `budgets()` |
| `resident_reference.ResidentReference` | 严格对照 | 在 GPU 上按相同 L、相同写入块重放同一请求序列，用来逐位比对 |

## 3. 所有权与不变量

- **互斥分层**：一个块只在 host 或 disk 之一。新块进 host；host 淘汰时溢写到 disk（失败则丢弃该块）；disk 淘汰时删除文件；disk 命中**不**回填 host。
- **host slab 只分配一次**：大小 = (slots + 1 staging) × 7,340,032 B，不超过预算。之所以不用 `pin_memory=True` 逐块分配，是因为 PyTorch 的 pinned 分配器会把大小向上取到 2 的幂，并缓存释放的块（D16）。
  - 从构造到 `close()`，整块 slab 一直占着，与占用多少无关。`resident_bytes` 只是 slab **内部**的逻辑占用，不是额外的内存。
  - 这些数字都不是 RSS 或页锁定的测量值。`metadata_bytes_estimate` 也只是估算，不是测得的 Python 堆用量（D21）。
- **pin / lease**：恢复中的 host 块 pins > 0，不会被淘汰。lease 在 `finally` 中释放，所以成功和异常都会释放。全部被 pin 时返回 `host_all_pinned`，跳过写入，绝不扩容。
- **staging 缓冲**：只有一个，同步使用。拷贝完成（`copy_` 同步）后才处理下一块。
- **GPU 侧**：恢复目标张量按块填充，只用已验证的前缀 `[0, L)` 构造 cache，因此不会留下部分无效状态。
- **磁盘目录**：只删除匹配自己命名规则、且位于根目录下一级的文件；遇到未知文件就拒绝使用该目录；`unlink` 不跟随符号链接。
  - **M1.1：先校验，再动手。** 用 `lstat`/`scandir` 检查，不跟随符号链接：
    - 根目录必须是真目录；
    - marker 必须是普通文件（符号链接、目录等一律拒绝）；
    - 其余条目必须符合命名规则，且是普通文件或符号链接。

    全部通过后才删除旧文件。marker 缺失时用 `O_CREAT|O_EXCL|O_NOFOLLOW` 新建；已有的 marker 绝不改写。读 chunk 时用 `O_NOFOLLOW`。
  - **M1.1：先删除，再归还预算。** `used = committed（可读条目）+ tmp 预留 + undeletable`。
    - `unlink` 抛 `OSError` 时，该文件转入 `undeletable`：仍计入预算和 `max_entries`，不可读，它的 key 也不会再写入（`undeletable_file_for_key`）。
    - 不自动重试；`clear()` 只做一次显式尝试。文件本来就不存在时，视为删除成功。
  - 适用范围：单进程、顺序调用、可信目录。这些检查不能防御并发或恶意的目录修改（TOCTOU）。

## 4. 失败分类（要点）

- **未命中**（记录原因并回退）：块不存在；文件缺失；截断或大小不符；magic/版本/dtype 非法；header 或 payload CRC 错；命名空间、key、parent、序号、tokens 不符；布局不符；读 `OSError`。
- **跳过写入**（不重算）：写或溢写时的 `OSError`（含 ENOSPC）；块超出容量；全部被 pin。
  - M1.1 新增：淘汰时删除受害文件失败（`evict_oserror:<errno>`）。受害文件转为 undeletable，本次 `put` 结束，不会接着淘汰其他条目。
  - M1.1 新增：可用空间被 tmp 预留和 undeletable 占满（`disk_budget_unavailable`）。此时直接放弃，不做任何淘汰。
  - M1.1 新增：tmp 清理失败。写入照样跳过，tmp 的预留**保留**为 undeletable。
- **未命中但无法删除**（M1.1）：坏条目的文件删不掉时，结果仍是未命中、回退重算；字节继续计入预算，该文件永不再读。
- **构造时拒绝**：目录不归我们所有，或格式不对（未知条目、marker 是链接或不是普通文件、名字像缓存文件但不是文件）；启动时旧文件删不掉。这些都发生在任何请求之前。
- **直接抛出**：host 里的身份不符（这属于进程内 bug）；模型 cache 的 dtype 或形状与 namespace 不符；非 `OSError` 异常；CUDA 错误。

## 5. 边界情况

- N 恰好是 C 的倍数（如 512）：块 [256, 512) 会被**写入**，但本次只**恢复** 256。后续更长的 prompt 可以用到它。
- N ≤ C：没有可恢复的块，完整 prefill；已有的块不会重复写入。
- 中间块损坏：只用它之前的块；损坏的文件会被删除；本次请求重算并重新写入该块。
- ENOSPC：块被丢弃，tmp 被清理，请求照常完成，前向次数不变。
- 文件删不掉（M1.1）：请求照常完成，前向次数不变（每个请求恰好 1 次 prefill + steps−1 次 decode，CPU 小模型和 GPU 测试都断言了这一点）。之后同一前缀会重算，因为那个块不再可读。

## 6. 你应该能回答的五个问题

1. 为什么相同的 256 个 token 出现在不同前缀后面时，必须得到不同的 key？如果 key 只哈希本块 token，会恢复出什么错误的 K/V？（提示：K 已经带有 RoPE 位置，而且 K/V 依赖整个前缀。）
2. `read_chunk_into` 为什么要在读取 payload 之前，用"期望的" namespace 检查 `payload_len`、`meta_len` 和 `token_count`？如果先按 header 里的长度分配缓冲，会出现什么问题？
3. 磁盘预算为什么要把 tmp 文件算进去？如果一次写入在 `os.replace` 之前失败，哪段代码保证 tmp 被删除、预算被归还？如果连 tmp 都删不掉，预算又怎么处理？为什么这样才不会超预算？（提示：`put` 的 `finally` 和 `undeletable`。）
4. 为什么 M1 的 GPU 验收用"GPU 常驻重放的逐位相等"，而不用"与完整重算在 atol 内一致"？对照 M0.1 的哪几个负对照能说明这一点？
5. M1_REVIEW 第 7 节里 4 个恢复请求与完整重算的贪心结果在第 4 步分叉，为什么这不说明存储有 bug？哪一项证据把存储因素排除了？
6. （M1.1）旧代码在 `unlink` **之前**就扣掉了预算。为什么这会让磁盘实际用量超过预算？为什么删除失败后，那个条目既不能再读、也不能自动重试？如果 marker 是指向外部文件的符号链接，旧代码会做什么？现在的代码在哪一步拒绝它？
