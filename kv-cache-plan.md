# KV Cache项目计划：Cost-Aware Tiered KV Cache for LLM Inference

目标：简历上与存储引擎互补的AI infra项目。差异化不在"做了三级缓存"（vLLM和LMCache都已有offloading），而在：
**在固定内存预算下，判断什么时候恢复缓存比重算更划算，并用实验和正确性验证支撑这个判断。**
三类有辨识度的成果：(1) 成本模型驱动的恢复/重算决策，用独立验证集检验决策质量；(2) 和重算对照的差分测试 + 缓存身份/损坏/缺失的安全回退矩阵；(3) 与多种基线在相同预算下的可复现对比。

面向AI infra、ML systems、FDE，兼顾SWE。
时间：4周，每周15–20小时（约70小时）。开始日期：____。
定位：可信、可复现的单机系统实验，不是复刻LMCache。核心是有界缓存、正确恢复、成本决策、对照实验；流水线是加分项，不是完成或写简历的前提。

## 权威文档
- docs/design.md（仓库内）是设计约定的唯一权威；冲突时以它为准，回来修本计划。
- 每周细节写在本文件，不再单独拆周计划。

## 开发环境
- WSL2（Ubuntu）+ 移动版RTX 4080（12 GB）；仓库放在~/
- 固定版本：推理框架、PyTorch、CUDA、驱动、模型revision，写进README和每次实验记录
- 模型：Qwen2.5-1.5B-Instruct（fp16；KV约28 KiB/token，16K上下文约450 MiB）；显存不够再降到0.5B
- 首版不支持：LoRA、多模态、自定义位置编码
- 主语言Python + PyTorch；只有确实写了C++/CUDA代码，技术栈里才写C++/CUDA
- GitHub Actions跑CPU可测部分（格式、身份、回退、预算、成本模型）；GPU测试本地跑，结果存档

## 核心约定（写进design.md）
- **缓存身份**：chunk key = 链式哈希（父chunk哈希 + 本chunk token ids）；namespace = 模型ID与revision、dtype、层数/KV头数/head_dim、RoPE配置、block大小、缓存布局版本、框架版本。身份同时写进chunk header，加载时逐项核对；不一致 → 未命中
- **只缓存完整chunk**（如256 token，按框架block大小对齐）
- **首token规则**：至少重算最后一个prompt token；恢复长度只能取 ≤ N−1 的完整chunk边界（N = prompt长度）。所有基线用同一规则，或明确披露差异
- **部分恢复**：恢复后的token位置、attention mask、缓存长度三者一致
- **失败分类**：
  - 已识别的缓存故障（文件缺失、截断、CRC错、格式非法、身份不符、读I/O错误）→ 未命中
  - 程序不变量违反、非预期异常、CUDA错误 → 抛出，测试失败；不用宽泛的except吞掉
  - 读取/校验失败 → 放弃受影响的恢复路径并重算；已验证且连续的前缀可以保留，否则清理本次恢复状态后完整重算，不留部分损坏状态
  - 写入/降级写盘失败（含磁盘满）→ 放弃这次缓存写入，照常返回已算好的结果，不重算
- **磁盘chunk格式**：header（magic、version、namespace哈希、chunk key、token数、payload长度、header CRC）+ payload + payload CRC；写入走tmp → rename，不fsync（断电丢失或零长文件由长度和CRC检查转为未命中）。对比存储引擎：持久数据fail closed，可丢弃缓存fail safe
- **容量有界**：
  - host预算 = 常驻chunk + 所有项目管理的staging buffer；NVMe预算 = 已提交文件 + 临时文件
  - 缓冲区数量和待写盘任务数量有上限
  - 恢复中的chunk加pin，不能被淘汰；所有候选都被pin时跳过本次缓存写入或放弃本次恢复，不临时突破预算
- **淘汰**：只做LRU（host满 → 降到NVMe；NVMe满 → 删除）。不做前缀感知淘汰，避免分不清收益来自决策、淘汰还是传输
- **数值校验口径**（写代码前定死，不在发现问题后放宽）：首token logits的atol/rtol、NaN/Inf检查、greedy前k个token一致率；阈值同时参考"重算 vs 重算"的波动和"部分prefill vs 完整prefill"的形状差异

## 4周计划
| 周 | 内容 | 验收标准 |
|---|---|---|
| 1 | 固定环境；接入推理框架；KV导出/导入；首token与部分恢复规则；差分测试；二维校准数据；可行性检查 | 恢复后KV字节一致；边界长度对照通过；有T_prefill(n｜L)网格和两层实际恢复时间；可行性结论记录 |
| 2 | host + NVMe两层缓存；chunk格式与身份；有界预算与pin；LRU；失败分类；"总是恢复"端到端 | 回退矩阵通过；预算从不超额；第一组TTFT与无缓存、GPU-only对比 |
| 3 | 成本模型；按成本选择恢复长度；决策质量验证；代表性配置对比，再在转折区域加密 | 决策一致率和决策额外耗时；四种配置对比表；结论如实记录 |
| 4 | P1：跨chunk流水线恢复 + profiler + 消融；P0：README、文档与最终验证 | README和简历数字；有流水线证据才写overlapping |

---

## 第1周：接入、对照、校准数据

### 需要实现
- 环境固定并记录；跑通一次推理
- 接入方式二选一并记decisions.md：
  - 首选vLLM的KV connector接口（固定版本，接口变化快，按该版本源码为准）
  - **最多投入6–8小时**，接不通就退回HF transformers（past_key_values），自己实现GPU-only前缀缓存当基线
- KV导出/导入：按chunk取出各层K/V → 放回 → 继续生成；实现首token规则和部分恢复
- 差分测试框架：同一prompt分别走"恢复"和"重算"
- 校准数据（二维，不只是一条曲线）：
  - 完整重算 T_prefill(N｜0)，多个N
  - 部分恢复后的续算 T_prefill(n｜L)，扫描多个(L, n)组合
  - 实际chunk恢复时间：host命中、NVMe命中各测一组，包括读取、校验、布局转换、GPU拷贝；裸带宽只作参考
- 可行性检查：粗看各层恢复成本和对应重算成本的关系

### 测试
- 字节级：恢复后的KV张量与存入时逐字节相等
- 端到端：按预先定好的数值口径对照重算
- 边界长度：chunk_size−1、chunk_size、chunk_size+1、多个chunk的边界；prompt恰好是chunk整数倍时，确认最后一个chunk不被恢复
- 部分命中：只恢复前若干chunk，其余重算，输出仍通过

### 注意事项
- 区分KV复制是否精确（应完全相等）和端到端数值是否等价（计算形状变化可能有浮点差异）
- 校准数据存results/calibration/，按配置分成校准集和验证集，第3周拟合只用校准集
- 可行性检查如果显示所有实际场景都适合恢复：及时把成果定位为"验证成本模型与性能边界"，不刻意寻找能让策略胜出的负载
- 本周不碰淘汰

### 完成定义
- [ ] P0 环境版本记录；接入方式决定记decisions.md
- [ ] P0 字节级恢复、边界长度、部分命中、端到端对照通过
- [ ] P0 二维校准数据和实际恢复时间入库，附生成脚本
- [ ] P0 可行性结论写进EXPERIMENTS.md
- [ ] P0 docs/design.md初稿：身份、首token规则、chunk、失败分类、预算、数值口径

---

## 第2周：两层缓存 + 安全回退

### 需要实现
- host pinned内存池：固定预算（含staging buffer）、chunk粒度分配、LRU、pin计数
- NVMe层：chunk文件格式、tmp → rename、目录预算（含临时文件）、LRU删除；host淘汰时降级写盘，待写任务有上限
- 查找：沿chunk链找最长命中，标注每个chunk所在层
- "总是恢复"策略：按首token规则恢复最长可用前缀，其余重算
- 计数器：各层命中/未命中、回退原因、读写字节、各阶段耗时（查找、读盘、校验、拷贝、prefill）、预算峰值
- 工作负载生成器：共享system prompt、多轮对话、RAG式长文档；每个负载记录请求命中率、命中前缀占prompt的比例、命中数据所在层、工作集大小与缓存预算之比
- 基线：无缓存；GPU-only前缀缓存（vLLM原生或自己实现），通过限制显存预算制造缓存压力

### 测试（结果写进docs/correctness-matrix.md）
- 格式解析大量放在CPU测试：逐字节截断、header/payload翻位、长度字段非法、身份字段逐项不符 → 未命中，无大分配
- 每类恢复故障再做代表性的GPU端到端验证：输出与重算一致
- 同token、不同模型/dtype/block大小 → 未命中，不误用；相同chunk内容、不同前缀 → 不同key
- 恢复到一半失败 → 保留已验证的连续前缀或完整重算，结果正确
- 写失败、磁盘满 → 跳过缓存写入，请求正常完成且不重算
- 不变量违反（如张量形状错误）→ 抛出，不被当成未命中
- 预算：任意时刻host和NVMe用量不超过预算；全部候选被pin时按规则跳过或放弃
- 恢复中的chunk不被淘汰；淘汰后再请求 → 从下一层恢复或重算

### 注意事项
- 回退路径和正常路径走同一套差分检查
- 预算在实验中固定并写进结果；看的是相同预算下少做了多少重复prefill，不把"容量从A到B"当成果
- 第一组TTFT只当基线记录，不急着优化

### 完成定义
- [ ] P0 回退矩阵通过：CPU格式测试 + 每类GPU端到端代表用例
- [ ] P0 预算测试通过，计数器记录的峰值从不超额
- [ ] P0 "总是恢复"在三类负载上端到端跑通
- [ ] P0 无缓存 / GPU-only / 总是恢复三组初步TTFT
- [ ] P0 BUGLOG有真实记录才写；decisions更新

---

## 第3周：成本模型与决策

### 需要实现
- 成本模型：
  - T_restore(L)：按第1周实测的chunk恢复时间（分层）建模，不只用裸带宽
  - T_prefill(n｜L)：用第1周二维校准集拟合
  - 选择恢复长度 L* = argmin over 可用chunk边界（≤ N−1）[T_restore(L) + T_prefill(N−L｜L)]，L = 0即完全重算
- 成本感知策略接入，和"总是恢复"共用同一套存储和计数器
- 决策质量验证（小规模、独立验证集）：对每个请求实测几个候选恢复长度的实际耗时，得到候选中的实测最优，再与模型选择比较
- 实验先做少量代表性配置：
  - 每类负载各一组典型参数，加数据大部分在host / 大部分在NVMe两档压力
  - 在发现的转折区域再加密扫描，不做全组合

### 对比的四种配置
1. 无缓存（全量重算）
2. GPU-only前缀缓存（相同显存预算）
3. 总是恢复（相同host/NVMe预算）
4. 成本感知恢复（相同预算）
- 可选参照：能跑通LMCache就加一组，标注为成熟实现参照，不预设要赢

### NVMe读取口径
- 分三类报告：项目host cache命中；host未命中但文件在OS page cache中；尽可能控制并验证的冷文件读取
- 冷读控制方法（如drop_caches或posix_fadvise DONTNEED）和验证方法（如检查文件是否驻留page cache）写进EXPERIMENTS.md
- WSL2下无法可靠控制宿主机缓存：不验证就不声称冷NVMe读取，结论限定为"WSL2下的文件恢复性能"

### 注意事项
- 成本决策的收益应出现在实际恢复成本高于对应重算收益的区域；不预设低复用率一定带来优势
- 可能不存在单一稳定的"盈亏平衡长度"；按区域报告模型的决策，而不是一个数字
- 如果实测没有收益：如实写进EXPERIMENTS.md，简历第二条改成验证过的其他收益（如缓存压力下相对GPU-only的改善）
- 统计口径：每组至少3次运行，每次运行分别算p50/p95，再报告各次p50/p95的中位数和范围；注明每次请求数；p99只在请求数足够时报

### 完成定义
- [ ] P0 验证集上：模型选择与实测最优的一致率、选错时多花的耗时（ms）、决策计算本身的开销
- [ ] P0 验证集上的时间预测误差（辅助指标）
- [ ] P0 四种配置的代表性对比表和图，原始数据入库
- [ ] P0 decisions记录模型形式和取舍；结论写进EXPERIMENTS.md（含不利结果）

---

## 第4周：流水线恢复（P1）+ 收尾（P0）

### 需要实现
- P1 跨chunk流水线：两个固定staging buffer；chunk i+1读盘与chunk i的host→GPU拷贝重叠；独立CUDA stream和事件同步
- P1 成本模型加入流水线修正（重叠后不能简单相加）
- P1 profiler（Nsight Systems或torch profiler）截取时间线；消融：流水线开/关 × 数据所在层
- P0 README、结果图、重现命令；文档与最终验证

### 注意事项
- 使用CUDA stream ≠ 已经重叠：没有profiler证据，就不写"overlapping"或"pipelined"
- 只做恢复流水线内部的重叠；和其他请求的计算重叠、批处理调度属于计划之外
- 时间不够时整块砍P1，前三周已构成完整项目

### 完成定义
- [ ] P1 profiler时间线截图入docs/；流水线消融数字
- [ ] P0 全部测试重跑通过；CI全绿
- [ ] P0 README：结果表、环境、预算、缓存控制方法、局限性、重现命令、AI使用说明
- [ ] P0 correctness-matrix、EXPERIMENTS、decisions完整；tag `v1.0`；简历换上实测数字

---

## 证据（从第1周开始积累）
- docs/design.md：身份、首token规则、格式、失败分类、预算、数值口径、成本模型
- docs/decisions.md：接入方式、chunk大小、模型形式、淘汰策略等取舍
- docs/correctness-matrix.md：场景、注入方式、预期、实际、测试名
- EXPERIMENTS.md + results/：每次实验的配置、原始数据、脚本、结论（含不利结果）
- docs/BUGLOG.md：只记真实bug
- 随手记录：差分测试prompt数、回退用例数、实验配置数

## 测量口径
- 所有对比使用相同模型、相同显存/host/NVMe预算、相同请求trace、相同首token规则
- TTFT从请求提交到首token；统计口径见第3周
- 计时前预热；不加调试同步（测量需要时注明）
- NVMe读取按三类分开报告；不把"引擎请求字节"说成物理设备I/O

## 计划之外
- 前缀感知淘汰、cost-aware准入（是否写入缓存）
- 保存额外输出状态，让全前缀命中时可以跳过最后一个token的重算
- 多请求并发下与计算的重叠、调度
- 多机/分布式缓存、KV量化或压缩、LoRA/多模态
- 用自己的存储引擎存元数据索引，实现服务重启后缓存可用

## 完成后的简历写法（数字测完再填）
Cost-Aware Tiered KV Cache for LLM Inference (Python, PyTorch, [vLLM])
- Built a tiered prefix KV-cache backend for [vLLM] with bounded pinned-host and NVMe tiers, [pipelined restoration,] and a calibrated cost model that chooses how much cached prefix to restore versus recompute.
- Reduced p95 time-to-first-token by X% versus GPU-only prefix caching under matched memory budgets across N workloads, and [result in the regime where the cost model mattered, named after the experiment]; the model matched the measured-best restore choice on Y% of held-out requests, with Z ms mean regret.
- Validated reuse with N differential tests against full recomputation and M injected faults; missing, corrupted, or incompatible (model/dtype/layout) entries safely degrade to recomputation.
（方括号内容按实测决定：没做成流水线就删"pipelined restoration"；成本决策没有收益就把第二条后半句换成验证过的收益。）
