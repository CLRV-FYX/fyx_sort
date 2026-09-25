# Changelog

## Unreleased — v10.4 candidate

Date: 2026-09-16

### Added

- **封顶自然归并前门（`try_proof_structured_sort`，工程名仍叫 PSS）**——
  **不是新排序范式**，而是自然归并 / Timsort 家族的受限制特化：一次封顶扫描
  （≤7 个违例位，第 8 个立刻放弃）找目标方向单调 run，然后标准稳定归并
  （r=2 回绕则 rotate）。与 Timsort 的工程差别只有：封顶拒绝（随机期望 O(1)
  退出）、r≤8 时固定二叉归并树（无栈）、rotate 特例、挂在 vqsort 调度链上。
  随机输入在头几个元素攒够封顶违例而放弃（串行零税实测 -1.4% 噪声带内）。
  串行入口仍只在 `!dynamic_parallel_allowed` 时触发；并行入口
  `try_proof_structured_sort_parallel`（n≥256K）分块扫 break + r=2 wrap
  走既有双缓冲 rotate，其余 fork_join 成对稳定归并。zigzag/bitonic 含反向
  run，不假装吸收。套件 743/0；1M int64 同块交错：concat2 par/ser 0.57、
  rotated 0.62、runs8 0.75、zigzag 1.02。**不是 canonical 矩阵。**
  串行 1M int64：2 段 -27.5%、3 段 -18.0%、8 段 -17.2%、concat2 -4.5%
  （3 轮同块交错中位），无任何回退。
- **单断点结构证明**（`try_one_break_rotate` / `try_one_break_rotate_parallel`）：
  有序性证明从 k=0（全单调才退出）推广到 k=1——前缀与后缀各自单调且端点回绕的
  范围是旋转有序数组，rotate 回来 O(n) 且精确；扫描在第二个违例处放弃（随机输入
  头几个元素攒够两个断点，实测串行零税）。串行版挂两处驱动；并行版为 chunk 缝隙
  代数（全 chunk 向量化单调扫描 + 恰一条坏缝/一个坏 chunk 精化 + 回绕检查），
  修复用全量双缓冲并行搬移。1M 并行 rotated：int64 -4%、int32 -6%、double -39%
  （对最强对手增益 5.4→8.7x）；8M 与 vqsort 并行平手（rotate 的带宽本质）。
  稳定路径（sort_st）用严格回绕条件守卫等值跨缝。

### Changed

- **串行向量快排 lean+max 分区**（`parts/10b_vsort.hpp`，仅 `vqsort_serial` 路径；
  并行步进分区 `vqsort_partition_step` 原样不动）。热循环不再跟踪区间 min/max
  （旧版每向量两条额外向量指令），改为只跟踪最大值（一条）：最小值与 `split == 0`
  检测冗余（低侧为空 ⟺ 枢轴是区间最小值，免费），最大值保留旧版的两个免费早退——
  整段与枢轴同值直接返回、高侧全为枢轴值时该侧已在终位、整体丢弃不再递归
  （重复合输入上每个此类子区间省两整趟）。同块交错 A/B（七轮中位，对 hwy vqsort）：
  int64 1M 0.82→0.85x、8M 0.84→0.86x；double 1M 0.87→0.89x、8M 0.85→0.86x；
  int32 持平（0.85→0.86x / 0.85→0.86x）。并行侧子任务（<64K 走串行内核）同享，
  并行 random 抽查无回退。
- 同轮否定并回退的实验：主循环软件预取（int32 受害，中性）、int32 采样 4×V（中性）、
  4 字节叶子 8 向量（更差）、分区 unroll=8（更差）、每槽 median-of-3 枢轴采样
  （int32 崩至 0.6x，标量加载 3 倍）。串行差距的下一批候选：hwy BaseCase 网络、
  PartitionRightmost 余数预处理、GatherSample 枢轴。

### Fixed

- **整数键的向量化证明变换用错**（v10.3 引入）：向量单调扫描对所有键类型套用
  IEEE 浮点变换（符号传播全翻转），该变换对有符号整数在负数域内**逆序**、对高位
  无符号值同样逆序——负 int64/高位 uint64 的单调数组被向量证明误拒（静默回落
  慢路径）。三处统一修复：无符号=恒等、有符号=仅翻转符号位、浮点=符号传播翻转；
  构造分析确认误收在现分发下不可达（无错误输出，仅性能）。教训：向量化键变换
  必须从 RadixTraits 按类型推导。
- 结构证明排序的向量化封顶扫描 off-by-one：第 7 个违例误触发封顶（应为第 8 个），
  8 段形状被误拒；修复为先查后写。
- **结构证明排序扫描向量化**：封顶违例扫描改 AVX-512（每 8/16 元素一次
  load+encode+移位自比较，掩码逐位抽取，封顶中途退出）。同头隔离 A/B（3 轮交错）：
  1M int64 r=3 证明路径 -13%、r=8 -14%（标量 encode 依赖链 ~2.5ms 是主要成本）。
- lean+max 首版的掩码尾部向最大值跟踪器喂了未掩码计数（垃圾通道被计为右侧），
  空隙簿记被破坏，右侧 compress-store 越过左界直至写出数组头（堆损坏 +
  乱序输出）。修复：数据移动仍走掩码变体，最大值更新单独对「垃圾通道换枢轴」的
  安全副本进行。教训：部分向量的垃圾通道不得进入任何计数或存储；探针的排序校验
  （rc=1 BAD fyx）与 ASan 分别在 -O3 与 -O1 抓住同一 bug 的两种表现。

### Result

- 串行随机算术列对 vqsort 本尊的差距从 13–20% 收窄到 10–16%（均匀随机，
  七轮同块交错中位）；正确性套件 743/0，ASan+UBSan 743/0，`-Werror` 干净。


## Unreleased — v10.3 candidate

Date: 2026-09-14

### Fixed

- **线程池外部线程竞态**（`parts/11_parallel.hpp`）：此前所有非池线程的
  `this_worker()` 都返回 0，N 个应用线程并发排序时全部假冒 worker 0 推/弹同一条
  Chase-Lev deque——单所有者协议被破坏，丢任务导致 `wait_for` 永转（8 线程压测
  2/20 挂死）。现在第一个外部线程认领 worker 0（粘性），其余外部线程走互斥 FIFO
  （空闲 worker 睡前抽干、外来等待者抽干）；热自旋路径不查 FIFO（同块 A/B 显示
  无条件检查让逆序家族慢 1.5–2.2x）。修复后 30/30 压测零挂死，TSan 无警告。

### Added

- **AVX-512 比较式小基数计数趟**（d≤16）：每键每向量块一次比较 + 一次掩码加，
  取代「乘→查表→依赖验证加载」链；未采样键表现为总数短缺而拒绝（`sum==n` 守恒
  保障正确性）。值域远宽于键数的形状（mod8 摊在大窗口上）改走 rank 计数，
  核开销 O(range×chunks) → O(n+d)。fill 按输出区间切分（计数倾斜不再单核写全数组）。
- **向量化有序性证明**（`radix_key_monotone_scan`，<2M 池辅助与 ≥2M 全并行两路共用）：
  每 8/16 元素一次加载 + 三条编码指令 + `alignr` 移位自比较 + 掩码测试，违例早退；
  位精确键保持浮点 NaN/-0 总序，与标量探测器等价（t_counting 对抗形状测试）。
- **向量化全等扫掠**（`range_all_equal_first_vec`，两路共用）：全等判定 = 每元素与
  p[0] 比较，免三向分类免接缝簿记；-0/+0 仍按编码键区分；异议元素回落原分类路径。
- **测量协议**（`tools/dev/merge_runs.py` + NOTES）：多轮独立进程逐格取中位数为
  canonical 口径；结论必须来自同块交错的新旧双二进制 A/B（运行块漂移可达 ±2x）。

### Result

- 标准模式记分牌：**1M 56 胜 / 0 负，8M 42 胜 / 0 负**（六对手 × 4 类型 × 14 分布，
  无任何一格低于 1.00x；最弱格 int64 远距离交换 1.01x）。上一版为 1M 53-3 / 8M 41-1。
- 关键翻身格：8M double 已排序 0.94x 输 → 1.34x 赢；1M int64/double 全等
  0.95x/1.07x → 3.17x/3.10x；8M 全等（int64/double）→ 3.8–4.0x；mod8 全家族
  0.61–0.80x 输 → 1.40–2.02x 赢。

## Unreleased — v10.2 candidate

Date: 2026-09-07

### Fixed

- **200 万以下的已排序/逆序/全等检查不再单线程空转**（`try_pool_assist_order_exit`）。
  此前排序本体在 1M 规模走串行时，有序性证明连线程池都不碰，IPS4o 并行的跨线程
  `is_sorted` 在这类平凡输入上把整个调用压到我们的 0.48x–0.65x。现在调度器在
  `kParallelProofMinN`(128K) 与 `kParallelOrderMinN`(2M) 之间、且 `n·sizeof(T) ≥ 3 MB`
  时，把 *证明扫描* 借给线程池而排序本体保持串行：只读、无分配，拒绝时调用方原路
  落回串行检测器。
  - 辅助路径不用 8M 的跨步抽样门：刚被写入的 1–8 MB 数组上，4096 个跨步探针是
    ~0.3 ms 纯冷未命中（1M double 已排序实测因此 +0.1 ms），比它把守的扫描还贵。
    改用顺序前缀（4096 元素串行分类，预取友好）：乱序输入微秒级拒绝且永不唤醒池；
    前缀单调才把单向验证（每元素一次比较、每 chunk 验证跨缝对、浮点编码键缓存）铺到池上；
    前缀全等才退回逐 chunk 分类 + 块缝合并。
  - 前缀+chunk 恰好覆盖整段，仍是完整证明而非抽样；裁决与调度记录和串行检测器逐位一致。
  - 合并逻辑的第一版把块缝错位一格并在 `p[n]` 越界读了一个垃圾对——只会错误拒绝、
    不会错误证明，但被新增的等价性测试（「全等前缀 + 严格递减尾部」形状）当场抓住后修正。
  - 实测（同机背靠背全量矩阵）：1M 已排序 int32 0.94x→1.52x、int64 0.65–0.83x→1.26x、
    double 0.53x→1.08x、string 0.55x→0.94x–1.03x（vs IPS4o 并行）；8M int32 1.04→1.22x、
    double 0.93→1.16x；decliner（随机/近似有序/旋转/拼接/块交换）1M 全部持平或更好。
    全量战绩 1M 46W/10L→**48W/8L**，8M 39W/3L→**41W/1L**。

### Verified

- **运行时 ISA 分发实测背书**：基线编译（无 `-march`）与 native 编译在 1M 随机/
  已排序全部格子性能持平（int32 随机 0.00253 vs 0.00251 s、int64 已排序 0.000203 vs
  0.000206 s）——库的 SIMD 内核（网络、基数、向量快排、画像扫描）经 `FYX_ISA_BEGIN`
  目标区域 + `use_avx512()` 运行时检查对**任何编译方式**自动生效，不要求用户懂得
  加编译参数（MSVC 无 per-function target，文档已注明例外）。

### Rejected

- **AVX-512 one-hot 窗口计数**（尝试过、测量否决、已删除）。假设 mod8 家族的瓶颈是
  标量计数链，向量化（每窗口值一次 cmpeq+popcount）能降 ALU；同树单变量 A/B 实测
  int32/int64 mod8 一律慢约 2 倍（1M int64 0.00147 vs 0.00067 s）。小窗口的计数表
  常驻 1-4 条缓存行、mod 循环模式天然错开自增依赖，标量已是强内核；one-hot 的操作数
  随窗口宽度倍增。向量化直方图只对宽窗口（256 槽 radix，已有 vpconflict 版本）成立。

### Changed

- **int64/uint64 高熵随机改配 AVX-512 向量快排**（`vqsort_preferred` 路由翻转）。
  旧决策是单机测量的产物：旧机器 8M 串行 radix 0.089 vs vsort 0.096（7%），本机
  同样的对决**反转为 vsort 快 2–3 倍**（1M 全域随机 0.0061–0.0094 vs 0.014–0.030），
  且 high-prefix radix 有病态 tie 案例（8M 40-bit 分布 0.73 s vs vsort 0.083，
  **9 倍**——top 前缀不唯一时 tie 修复趟的代价）。最坏情况 -7% 的默认无法对抗
  最坏情况 -90% 的默认：路由在所有主机上统一翻转向量快排，radix 家族保留它独占的
  形状（置换范围、前缀守卫计数）。矩阵规范结果：int64 随机 1M 0.44–0.55x→**1.34x**、
  8M 0.75x→**1.57x**（对 vqsort 本尊）；**8M 战绩 41 胜 1 负**——除一个坏窗口格外
  （int32 已排序 0.87x，相邻轮 1.2x+），8M 对全部六个对手每个格子都获胜或打平
  （vqsort / xss / IPS4o 串行 / pdqsort / std 均 42/0）。稳定性契约不受影响
  （stable 路径走 stable_merge_sort，从不进 vsort）。嵌入套件 743/0、t_counting
  `-Werror` 干净、int64 全形状探针（全域/负值/40-bit/20-bit/极值/降序）通过。

### Changed

- **镜像反转铺到线程池**（`reverse_range_adaptive(p, n, parallel_swap)`）。反转是
  纯带宽操作，镜像对 (i, n-1-i) 相互独立；当调用方已获准用线程（有序性证明已经
  并行验证）且数组 ≥3MB 时，交换按对空间分段并行。大 n 时 `try_fast_reverse_exit`
  让位给「并行证明 + 并行交换」两趟。本机 2 核带宽封顶：int32 逆序 8M 稳定改善
  （4.8ms→3.4ms），int64/double 持平不回归；核更多的机器随真实带宽继续放大。
- **密集窗口计数扩展到全部 radix 类型**（按编码键）。浮点编码键是保序整数，
  「编码值落在窄窗口」的 double 列（评分、定点金额、`i % 8`）现在与整数共用
  O(n+窗口) 的 u32 计数器，不再走哈希探测的稀疏路径；窗口外键立即逃逸回精确路径。
  1M double 逆序转赢、double mod8 0.40x→0.72x、int64 mod8 0.44x→0.66x（vs 小值域专精）。
  **IPS4o（串行+并行）两个规模全部败格清零。**

### Changed

- **低基数计数路径去掉了一整趟与所有冗余读**（跨环境成立的结构性精简，`SampleWindow`）。
  输入画像（profile）的 1024 点抽样现在把「已编码键的 min/max 窗口 + 精确 distinct 数」
  随 `InputProfile` 一并携带，串行/并行密集计数、串行/并行稀疏计数全部改为消费这份
  窗口证据，不再各自重读一条 1024 点跨步抽样（每条都是一排冷缓存行）；计数开始前
  也不再需要全量 min/max 扫——直接按窗口计数，任一键落在窗口外立即逃逸回精确路径。
  串行密集计数从 5 趟（画像、稀疏检测抽样、minmax 抽样、全量 minmax、计数+回填）
  减到 3 趟；计数表从 `size_t` 换成 `uint32_t`（表流量减半）；稀疏计数在样本
  distinct ≤ 24 时用 64 槽小表（与数据流同驻 L1，探测链消失，饱和即让位给大表）；
  并行稀疏内核删掉对每个 chunk 17 KB 表的冗余 `fill(0)`（vector 值初始化已保证全零）。
  该家族（状态码/枚举/类别——每个领域的最常见形态）在缓存更紧张的机器上受益更大。
  本机 ±50% 噪声下逐格不可分辨，交替复核无回归信号（对照格同样摆动）；正确性由
  嵌入套件（743/0）与 t_counting（计数内核全覆盖、窗口逃逸、-0/NaN 全序）保证。
### Added
- `test/t_vsort.cpp`: 7044 checks over the new kernel -- 10 shapes x 17 sizes x
  6 types, kernel and entry point, ascending, descending and stable, plus the
  floating-point edge cases (a range holding a NaN or a -0 must make the kernel
  step aside, and the result must still be the documented total order).
- **AVX-512 vectorised quicksort** (`parts/10b_vsort.hpp`), the kernel the
  radix accounting in `tools/dev/NOTES.md` said was the only route left on
  high-entropy numeric data.  Three passes of LSD radix on int32 is ~9.4
  ns/elem of CPU work and no cheaper radix exists (blocking, wider digits, a
  deeper write-combining buffer and a vectorised scatter were all measured and
  lost); this replaces the passes with an in-place `vpcompressd` partition,
  four vectors per iteration, the side to read chosen with a cmov rather than
  a branch, and a leaf of up to 16 vectors run through the existing Batcher
  network over native values (no encode/decode).  1M random, parallel:

  | | before | after | vqsort |
  |---|---:|---:|---:|
  | int32  | 0.0048 s | 0.0025 s | 0.0033 s |
  | double | 0.0068 s | 0.0054 s | 0.0066 s |
  | int64  | 0.0055 s | 0.0051 s | 0.0075 s |
  | float  | 0.0122 s | 0.0032 s | -- |

  8M random parallel: int32 0.038 -> 0.030, double 0.066 -> 0.054.  The random
  family, which was the one systematic loss against vqsort (0.79x at 8M), is
  now won at every size and type measured.  Floating point declines the kernel
  when the range holds a NaN or a -0, whose total order a hardware compare
  cannot reproduce; those keep the radix path.  64-bit integers keep the
  high-prefix radix, which is still faster for them.

### Changed
- Dispatch: three measured corrections, all of them ranges that were going to
  a kernel that is no longer the fastest available.
  - The high-prefix radix degenerates when the prefix is nearly constant -- 4M
    int32 drawn from a 2^22 range took 0.111 s through it, against 0.019 for
    the quicksort.  The quicksort is now reachable for every key span.
  - A narrow value range is not the same property as a low value count, so the
    range counters were unreachable for high-entropy input: 4M int32 over a
    2^10 range cost 0.049 s and now costs 0.0052.  Both range kernels self-gate
    on range against n, and hand the wide half to the quicksort.
  - The patch merges are sequential and move every element at least twice, so
    a pool beats them once the dirt is real: 8M int32 with 0.1% of positions
    swapped, 0.0454 s -> 0.0292.  They keep everything lighter than that.
- Wide-range, few-value input goes to the sparse counter instead of the dense
  one: 16 values spread over 61440 (1M int64 lowcard16) cost 0.0021 s parallel
  and 0.0031 s serial, against 0.0013 / 0.0022 for the sparse kernel.
- Organ-pipe/zigzag detector: the two full-range shape scans (prefix must be
  non-increasing, suffix non-decreasing) now run as one fused pass over the two
  ranges instead of two sequential scans.  Semantics unchanged; halves the scan
  loop overhead on shapes that reach the full check.  8M double zigzag (the
  last big comparison loss at 8M, 0.92x vs pdqsort) goes to ~1.5x vs pdqsort
  in paired runs; 1M double zigzag ~1.9x-2.0x from ~1.3x.  int32/int64 zigzag
  also improve.
- **Parallel orderedness proof** (`try_parallel_fast_order_exit`, reached above
  `kParallelOrderMinN` = 2M elements when the caller allowed threads).  The
  serial detector is one thread's O(n) scan, and on trivial input that scan *is*
  the runtime: 8M already-sorted doubles spent 0.0032-0.0042 s proving order and
  0.0004 s on everything else, so IPS4o -- whose sorted check is a striped
  `std::is_sorted` over TBB threads -- beat the whole call 0.48x-0.60x.  The
  proof is now split, and it matters *how* it is split:
  - a 4096-point strided sample gate decides direction; if it sees neither
    direction the range declines in microseconds, which is what keeps this from
    taxing non-monotone shapes (random, rotated, blockswap: 3-5 us);
  - when the sample both found a direction and witnessed a strictly ordered
    pair, the range is monotone-or-nothing and the sampled direction is the only
    answer that can be true, so each worker validates exactly that -- one
    comparison per element, each chunk re-checking the pair across its seam, so
    touching every element is the whole proof.  Classifying each chunk instead
    (`detect_fast_order_kind`, which is built to answer *either* direction, and
    on 2M+ strings per 512K-element chunk) costs two comparisons per element and
    was the difference between a bandwidth-bound scan and one that cannot keep
    the prefetcher fed: 8M int64 0.0054 s -> 0.0034 s, 8M int32 0.0037 ->
    0.0029 s.  Floating point keeps the previous element's *encoded* key rather
    than re-encoding both ends of every pair (two encodes per element measured
    3x slower than the loop it replaced).
  - only ranges the sample saw as all-equal fall through to the chunked
    classification, which preserves the AllEqual verdict and the memcmp
    shortcut for genuinely all-equal input.
  Net against IPS4o parallel on 8M: int32 0.60x -> 1.06x, int64 0.48x -> 0.98x,
  double 0.65x -> 0.97x (all three now at parity or better, and vqsort/x86-
  simd-sort are 10-20x behind on this shape).  Threshold is 2M on purpose: at 1M
  the pool wake-up costs more than it saves (measured +8-14% on the 1M
  nearly-sorted row) and the 1M string scan gains nothing measurable from the
  split.  `test/t_counting.cpp` now has two blocks: the parallel and serial
  proofs are checked for *identical* verdicts and identical decisions over
  1M/2M/4M x 8 shapes x {less, greater}, including shapes built to fool the
  sample (monotone at every sampled position and inverted in between, constant
  at every sampled position and not in between), plus end-to-end large-n sorts
  that assert the parallel exit is what proved the sorted and reversed cases.
- `bench_matrix`: vqsort/xss cells on `std::string` rows now print `n/a` and
  are excluded from the best-other choice -- a no-op vqsort call previously
  timed ~0 and hijacked the ratio column for every string row.  The file header
  now warns that `--filter=` runs consume a different prefix of the shared rng
  stream and therefore generate *different data* than the same row inside an
  unfiltered run; only numbers from identical command lines are comparable.
### Fixed
- **Wrong output order** (not just slow): `parallel_sort_ptr`, the task-parallel
  fallback, sorted its two halves with the library's order -- for floating
  point that is the radix total order, `-NaN < -inf < ... < -0 < +0 < ... <
  +inf < +NaN` -- and then merged them with the raw comparator, which compares
  NaN with `<` and gets false both ways.  70000 floats holding NaNs came out
  with three inversions, on this version and on every earlier one.  The merge
  now uses the same order the halves were built in.  Found by the new
  `test/t_vsort.cpp`, which checks the documented total order after every sort.
- A pivot equal to the range minimum partitioned into an empty low side, so a
  range where one value owns more than half the elements walked its whole
  depth budget before falling back to pdqsort.  It is re-partitioned with the
  strict test now, which always moves at least the pivot-valued block: 2M
  int32 that are 90% `INT32_MIN` sort in 0.0019 s.
- `FYX_GPU_COMPUTE` never compiled: the CUDA kernel source is assembled from
  raw string literals opened with `R"CUDA(` and closed with `)"`, which is not
  the matching delimiter.  It builds now (still unverified at runtime -- there
  is no GPU here), with and without exceptions.
- Six build configurations that did not compile: `FYX_DISABLE_SIMD` (a known
  defect, listed in the README as unfixed -- the spin hint and the store fence
  are not vector kernels but live in the SSE headers), `FYX_DISABLE_PARALLEL`
  (the moving merge used by the sequential stable merge sort sat inside the
  parallel guard), `-fno-exceptions` (the C ABI wrappers had an unguarded
  try/catch), a plain `-fno-exceptions` without `FYX_NO_EXCEPTIONS` (GCC and
  Clang leave `__cpp_exceptions` undefined rather than defining it as 0, so the
  feature test never fired), and `FYX_USE_PDQ_PARTITION=0` /
  `FYX_SAMPLE_SORT_V2=0` (-Werror on an unused parameter and an unused
  typedef).
- Two `-Werror` warnings that broke *every* build of the shipping header at
  `-Wall -Wextra -Werror`: an unused variable in the natural-run merge and an
  unused typedef in the profile-pattern pdqsort.

## Unreleased — v10.1 candidate

Date: 2026-08-27

### Added
- `fyx::sort` accepts containers whose storage is not one block.  `std::deque`
  and `std::list` used to be refused (`list` would not even compile): a
  container with a `.data()` keeps taking the contiguous kernels, one with its
  own `sort()` (the node-based containers) keeps splicing -- faster than moving,
  and nothing a buffer could beat -- and anything else is moved into a buffer,
  sorted by the same kernels a vector gets, and moved back.  A segmented range
  also used to have `Options` silently dropped, so asking for parallel cost
  exactly what not asking cost.  1M int32 in a `std::deque`: 0.051s before,
  0.026s now, against 0.086s for `std::sort`.
- `tools/fyx_test.py` -- one portable file holding the whole test suite.  It
  writes the C++ out, finds the header and a compiler, builds, and runs:
  `python3 tools/fyx_test.py [--header P] [--compiler clang++] [--bench]
  [--rlimit-mb N]`.  743 checks: 17 shapes x 6 types serial and parallel, sizes
  from 0 to either side of every dispatch threshold, containers, comparators,
  and stress -- inconsistent and non-transitive comparators, throwing
  comparators and throwing moves (checked for lost or duplicated elements),
  reentrant sorts from inside a comparator, eight threads sorting at once,
  repeated sorts of the same input, move-only payloads, NaN / -0 / +0 /
  denormals, and a memory-cap probe that runs under `ulimit -v` to prove the
  degraded paths work when scratch, buffers and worker threads cannot be had.

- Low-cardinality counting paths:
  - dense integer range counting;
  - sparse integer counting for up to 256 encoded keys;
  - compressed comparator-equivalence counting for arbitrary payload objects;
  - `std::string` value counting for default ascending/descending order;
  - guarded trivial-prefix field counting for common `struct { int key; ... }` custom comparator shapes.
- Parallel specializations for the low-cardinality paths:
  - parallel sparse integer/radix-key counting, including default-order `float`/`double` low-cardinality keys;
  - parallel trivial-prefix field counting/scatter;
  - parallel top-level `std::string` MSD byte-radix sorting.
- `std::string` MSD byte-radix sort for default string order, including descending support via reverse.
- Parallel sample sort over the existing Chase-Lev work-stealing pool.
- IPS4o comparison harness: `bench/bench_ips4o_compare.cpp` and documentation in `bench/README_ips4o_compare.md`.
- IPS4o regression matrix now includes a high-distinct `struct { int key; int payload; }` custom-comparator case.
- Adaptive-order weapons (`parts/12b_adaptive.hpp`), three general structure detectors that replace per-shape special cases:
  - natural-run merge — one scan finds the maximal monotone runs, descending runs are reversed in place and the runs are merged bottom-up with a buffer no larger than the smaller side, so rotated arrays, concatenated sorted blocks and block permutations cost one or two passes instead of 4-8 radix passes;
  - local dirty-patch merge — positions taking part in an adjacent inversion are pulled out, sorted, and merged back over the compacted clean run;
  - displacement-patch merge — element `i` may stay iff it is `>=` everything before it and `<=` everything after it; two scans compute that exactly (two bits per element, no iteration), so blocks and segments moved anywhere are found too, including shapes the adjacent-inversion test cannot see at all.
  All four are read-only until the structure is proven, only permute, and keep the radix total order for default-order floating point.  `FYX_ENABLE_ADAPTIVE_WEAPONS=0` compiles them out.
  - sorted-affix sort — longest non-decreasing prefix, longest non-decreasing suffix, sort the stretch between them, merge the three runs.  A sorted table with a batch of new records appended, a log with an unflushed tail, a file with one damaged region: 1M int32 whose last tenth is shuffled was 0.031s through eight radix passes and is 0.0039s here, against 0.0046s for pdqsort.  Both scans walk inwards and stop at the first inversion, so a range with no ordered head or tail costs two comparisons.
- `test/t_adaptive.cpp`: the three weapons over 13 shapes x 4 types, serial and parallel — result matches `std::sort`, a declining weapon leaves the array byte-identical, high-entropy input is rejected cheaply, NaN/`-0`/`+0` keep the radix total order, custom comparators and odd sizes included.
- `bench/bench_matrix.cpp`: local distribution matrix (int32/int64/double/string x 16 distributions) with optional ips4o/pdqsort/`std::sort` reference columns, copy time excluded.
- Counting/string/comparator-key/parallel-merge edge-case tests in `test/t_counting.cpp`.
- Unified top-level input profiling (`InputProfile` / `profile_input`) with dispatch test hooks for sorted, reverse, all-equal, low-cardinality, partially sorted and high-entropy cases.
- Contiguous iterator fast-path detection for vector/string-style normal iterators, so `fyx::sort(v.begin(), v.end())` reaches the same pointer dispatcher as `fyx::sort(v)`.

### Improved

- Adaptive repair for order that already nearly holds (`v10.1` weapon-eight work):
  - bounded insertion repair now decides from the disorder itself instead of from a sample: insertion sort's cost is the inversion count, and an array whose 128-element blocks were permuted twenty times has thirty-eight inversions in a million positions, none of them in the first four thousand.  The repair polices its own budget while it runs — long travel is refused (an element crossing an eighth of the range is a block move), the running rate is capped with a strike to spare (shifts arrive in bursts), and the absolute budget is two shifts per element.  1M int32 block swaps 0.011s -> 0.0027s.
  - it rehearses on a copy of the prefix first, because insertion sort permutes as it goes and a shape abandoned half-way through used to come out the other end no longer recognisable to the detectors downstream.
  - the density gate that used to refuse nearly-sorted input (it required 8 inversions per 100 ordered pairs) is gone; the shift budget does that job and measures the quantity insertion actually pays for.  1M nearly sorted: int32 0.0049s -> 0.00057s, int64 8.9x faster than before, string 0.031s -> 0.0076s.
  - merging a small patch into a large run now touches the run in blocks: compaction walks the dirty bitmap a word at a time, and the merge gallops back through the run to move in one go everything that belongs after the next patch element.  1M int32 far swaps: patch merge 0.0079s -> 0.0040s.
- The high-prefix radix pass picks its width from the data instead of from the type.  A pass costs two traversals per digit plus a tie repair inside every prefix group, so the width that pays depends on how many groups the range falls into, and that is a property of the data: doubles in a narrow range share their high bits and collapse into tens of thousands of groups at 24 bits, where the tie repair costs more than the extra pass a 36-bit prefix would have spent, while uniform 64-bit data splinters into millions of groups at 24 bits already and there the wider prefix buys nothing but two more traversals.  A 4096-element sample estimates the group count (S samples in G groups collide about `S*S/(2G)` times) and keeps the narrow prefix while the groups it leaves average two elements or fewer.  1M double, uniform bits: 0.0245s -> 0.0186s serial, 0.0147s -> 0.0118s parallel; 1M int64 in [0, 2^50): 0.046s -> 0.0152s parallel.
- Comparison sample sort now checks for already-ordered, reversed and single-value input itself, so callers that bypass the profile (recursive bucket sorts, a comparator the profile cannot see through) no longer pay for sampling, splitter search and 256-way classification.  The pass abandons itself as soon as the range can be none of the three, so random input leaves after two or three comparisons.  Calling the sample sort directly on 1M int32: sorted 0.0097s -> 0.0012s, reverse 0.0184s -> 0.0012s, all equal 0.0036s -> 0.0009s.
- Monotone distribution detection for sorted/reverse/all-equal inputs is now centralized in the weapon-seven profile layer and fronted by `FYX_ENABLE_FAST_PATHS`: arithmetic all-equal uses shifted `memcmp`, large string all-equal can validate in parallel, sorted inputs take a single proof scan, reverse inputs can verify while swapping, front-reversed organ-pipe zigzag (the bench_final shape) is detected before the reverse fast path and sorted by reversing only the first half, dense adjacent-swap zigzag now uses an in-place pair repair, local sawtooth zigzag uses a sample-budgeted bounded insertion repair, half-organ/bitonic zigzag verifies two consecutive monotone runs (with direct arithmetic fill for consecutive numeric domains and a half-buffer string interleave for the fixed-width organ shape), and broader interleaved zigzag inputs are recognized as two parity monotone runs; concatenable runs use a half-buffer deinterleave before falling back to the full linear merge.  The profile samples 1024 elements, then when worthwhile performs one linear validation scan that combines sorted/reverse/all-equal checks, capped distinct detection and adjacent inversion counting.  Low-cardinality samples are used as a dispatch hint and are still fully validated by the existing counting paths, avoiding an extra profile-only O(n) pass on the low-cardinality hot path.  Default-order floating point uses radix keys to preserve NaN and `-0/+0` semantics.
- Partially sorted inputs whose adjacent inversion count is at most `n/64` now first try adjacent/local repairs; numeric default-order long-distance nearly-sorted inputs first detect consecutive integer / integer-valued floating permutations and fill the ordered range directly, then fall straight into radix after local repair declines, avoiding the old profile+pdq path; non-numeric long-distance perturbations use an iterative dirty-patch merge before pdq fallback.  Floating-point default-order pattern repair/pdq uses radix-key comparison so NaN and `-0/+0` total-order guarantees are preserved.
- Auto parallel dispatch now has a dynamic minimum-size gate (default about one million elements, plus a 128 KiB/thread floor) that can be overridden with the `FYX_MIN_PARALLEL_SIZE` environment variable, reducing small-input task overhead without affecting explicit `Tri::On`.  1M-scale 32-bit wide-radix now uses 1 chunk/worker and disables NT stores before returning to the larger-input WCB/NT path, and guarded natural-order comparator recovery gets serial high-prefix radix for high-entropy 64-bit integers and `double`.
- Radix-key sparse counting now covers all radix-supported scalar keys, not just integral types; this gives default-order low-cardinality `float`/`double` inputs an O(n) count/fill path while preserving radix total-order semantics for `-0/+0` and NaNs. Large low-cardinality radix-key inputs also have a parallel count/fill variant, and the tiny distinct-table hash was shortened to reduce per-element overhead on low-cardinality numeric data.
- Sample sort now uses a lighter IPS4o-style oversampling policy and temp scatter for non-trivial object types to improve cache locality.
- Sample-sort permutation was replaced with block-level bucket reorder: per-source-block bucket counts/prefix bases scatter into a temporary buffer, then copy/move back by block.
- Comparator-key sorting for trivially copyable records now detects integer key fields (`int32/uint32/int64/uint64` prefixes/offsets), validates comparator semantics by sampling, uses low-cardinality count sort when appropriate, and otherwise uses payload-preserving field-key radix sort with a final comparator `is_sorted` guard.
- High-distinct comparator-key probes now skip expensive low-cardinality counting attempts before entering field-key radix sort.
- Parallel divide-and-conquer merge now prefers buffered recursive block merge and falls back to `std::inplace_merge` on allocation failure or unsupported shapes.
- High-entropy numeric default-order inputs now enter chunked parallel radix beyond the 64-bit-only case; 32-bit integers and `float` can use the same multi-chunk count/scatter schedule, the first radix pass reuses the initial local histograms, integer/`float` value-buffer radix avoids the extra encode/decode arrays when profitable, and the chunk scheduler uses more fine-grained chunks for better 4-core/VM load balance.
- High-entropy radix now has a safe all-pass fast plan: if an evenly-spaced sample proves every radix byte varies, the parallel path counts only byte 0 for the first scatter instead of building a full all-pass planning histogram.  If any byte looks degenerate in the sample, FYX falls back to the full histogram/skip-pass planner.
- Added an MSD-bucket hybrid for the remaining vqsort numeric gaps, but gate it to 2-worker/bandwidth-constrained runs after the 4H8G matrix showed normal chunked radix is better on four hardware threads.  The hybrid splits high-entropy `float` on the top radix byte and 64-bit integer/`double` on the top 16 radix bits, then sorts buckets with tiny-bucket PDQ or lower-pass radix.
- Profile-hinted sparse low-cardinality numeric inputs now use sample-gated parallel dense-range counting for 64..65536-wide integer domains before falling back to sparse/radix-key counting, avoiding a wasted full min/max scan on huge-span sparse data while speeding up 256-way integer lowcard.  Floating-point plus sparse 32/64-bit low-cardinality keys also get a collision-free rank16 direct-map counter before the hash-table sparse path, with dense integer ranges and the compact float/double prefix direct-map kept ahead of it; the hash-table radix-key fallback now also keeps the full 256-way cap for `float` instead of the old 64-way guard, so arbitrary 256-way float lowcard cannot fall through to radix just because the rank16 projection is unlucky. This turns arbitrary 256-way float and wrapped/sparse 32-bit integer lowcard into near pure count/fill passes while preserving exact radix-key validation.
- Chunked parallel radix now uses 32-bit per-chunk histograms/recount buffers when chunk sizes fit, and chooses coarser chunks for floating/wide-key numeric sorts.  Recount passes use banked 4-way local counters; 64-bit integer value-buffer radix is restored to the measured 3 chunks/worker setting after the 2 chunks/worker experiment regressed the 4H2G vqsort matrix, and the `double` key-buffer radix path decodes during the final scatter into the user array without entering the integer value-buffer instantiations.  This reduces store-forwarding pressure, cache footprint and the extra output-copy cost without changing the public API or low-cardinality dispatch.
- High-entropy 32-bit integer random inputs now try a three-pass 10/11/11 wide-key radix before the byte-wise fallback, cutting one full count/scatter round and using 1M-aware chunk/NT-store gating to reduce task and WCB pressure.  High-entropy 64-bit default-order random inputs try a guarded high-prefix radix before the full 8-pass fallback: `int64`/`uint64` use a two-pass top24 prefix at 1M scale and top26 beyond that, while `double` uses top36 at 1M scale and keeps top39 for larger ranges.  The high-prefix tie repair scans decoded output with one cached prefix per element instead of re-encoding boundary elements twice.  A prefix-distinct sample gate keeps narrow-range/low-prefix data on the existing full radix path, preserving low-cardinality wins and correctness.
- Custom comparators that sample as natural ascending/descending order now safely recover the radix/counting fast paths for arithmetic types and the high-prefix radix path for high-entropy 64-bit numerics; partially-sorted natural-comparator numerics also probe this guarded radix route before pdq/patch fallback.  Low-cardinality `std::string` comparator inputs now use unordered value-count/fill regardless of comparator type, and large inputs use sampled exact keys plus parallel count/fill while sorting only the distinct strings; guarded MSD remains available for natural-order string random data.
- Sample-sort classification now uses an unrolled fixed-256 Eytzinger descent for cheap/trivial payloads while keeping the compact looped classifier, previous 128K recursion handoff, and block scatter for `std::string` fallback paths. Non-string serial sample-sort scatter uses a single prefix-position pass. Parallel arithmetic comparator fallback now uses a 64-way top partition with the old 256-way sampling budget, cutting random numeric classification from eight to six comparisons per element while preserving the 256-way low-cardinality signal. Low-distinct arithmetic samples use a tiny exact counter before falling back to pdqsort; the floating-point duplicate gate now also tries a collision-free rank16 exact counter (then the sparse hash counter as fallback) up to the 256-way sample band instead of sending 256-way float/double comparator inputs straight to pdqsort. The comparison recursion threshold remains lower for non-string data to keep high-distinct buckets in sample-sort longer when that is cheaper than large pdqsort leaves.
- Degenerate sample-sort splitter cases now fall back to pdqsort instead of recursing without progress.

### Changed

- The high-prefix partition — two passes over the top bits, ties finished while
  the buckets are still in cache — used to be reachable only for 8-byte keys,
  so random `int32` fell through to the 32-wide sort, which was 1.6x slower than
  even the plain LSD sort it was meant to replace.  The helpers now take their
  shifts from `sizeof(Key)`, 4-byte keys get their own `radix_choose_prefix_bits`
  branch ((24,12) up to 2^21 elements, (26,13) above), and both dispatchers try
  the high-prefix sort first.  Parallel `int32` routes through the parallel
  high-prefix kernel as well; the two parallel kernels cross over between 2M and
  4M, so the 4-byte branch declines past 3M.  Random data, 2 vCPU: int32 1M
  parallel 0.0098 -> 0.0061, int32 8M serial 0.154 -> 0.083, 16M parallel 0.080.
- Tie repair no longer pays a full sweep to find the groups it is about to
  repair.  AVX-512 compares sixteen elements (eight for 64-bit keys) against the
  vector rotated by one lane and produces a mask with a bit wherever an element
  continues its predecessor's group, so a vector with no ties is skipped whole
  and one with ties is walked a group at a time.  Tie scan on 1M int32 without
  ties: 2.23 -> 0.21 ns/elem; on 8M: 1.36 -> 0.53.
- Scatters are now chosen per size instead of per kernel.  The write-combining
  scatter — one cache line per bucket, flushed with a non-temporal store —
  keeps a large sort out of the read-for-ownership business, but its bookkeeping
  costs about 3.3 ns/element, and an array that fits in cache has nothing to
  save.  Below roughly 8 MB of keys the wide high-prefix kernel uses a
  conflict-detection scatter instead: `vpconflictd` for the lanes that want the
  same bucket, `vpopcntd` for each lane's rank among them, a gather for the
  bucket's running position, and two scatters to write the keys and the advanced
  positions back.  `tools/dev/scat.cpp`, 12-bit digits, 1M: 2.74 ns/elem against
  3.26 for write-combining; at 8M the two switch places (4.57 against 3.14),
  which is where the threshold sits.  End to end, 1M random: int32 parallel
  0.0061 -> 0.0049, double parallel 0.0118 -> 0.0077; 8M unchanged.

### Added

- `tools/dev/{scat,hist,correct,timer}.cpp` — the measurements the last two
  changes rest on, so the next round does not have to rebuild them: scatter
  kernels against each other across sizes, histogram variants, a radix
  correctness harness that straddles every crossover (sortedness and equality
  with `std::sort`, serial and parallel, 6 int32 shapes, every key width), and a
  serial-vs-parallel timer.  `hist.cpp` is why the AVX-512 conflict histogram in
  `09_radix` stays unwired: 0.72 ns/elem against 0.57 for the shipping four-bank
  scalar one.

### Fixed

- Move-only payloads (`std::unique_ptr`) now sort.  The parallel merge assigned
  through a const lvalue, and the sample sorts keep copies of the elements they
  sample: both refused to compile for a type that cannot be copied.  The merge
  moves now, and a payload that cannot be copied is routed to the comparison
  sort or the task-parallel divide, which only move.
- Running out of memory no longer propagates out of `fyx::sort`.  The fast paths
  allocate -- scratch, buffers, worker threads -- and `std::sort` never does, so
  a sorter whose fast paths do inherit that failure is strictly less robust than
  the sort it replaces.  `bad_alloc` and a worker that cannot be started now fall
  back to the in-place comparison sort, which needs neither.  Verified by sorting
  2M int32 under a 60 MB address-space cap.

### Benchmark snapshot

Google Highway 1.4.0 is in the tree, so the matrix has a third competitor next
to `std::sort` and `pdqsort`: `vqsort`.  `bash tools/dev/vqsort.sh 1000000` (or
`8000000`) clones Highway, builds the matrix and writes `build/vqsort_<n>.txt`;
`python3 tools/dev/mkbench_md.py` turns those files into `BENCHMARKS.md`, which
therefore contains no hand-typed numbers, including the cells we lose.

    arithmetic cells, parallel fyx::sort against the best of the three
                                    1M            8M
      wins / losses              32 / 10       38 / 4

    random, parallel             fyx      vqsort
      1M  int32  0.00520      0.00371      0.71x
      1M  int64  0.01282      0.00781      0.61x
      1M  double 0.01723      0.00819      0.48x
      8M  int32  0.05433      0.03711      0.68x
      8M  int64  0.06322      0.08140      1.29x
      8M  double 0.07984      0.08783      1.10x

Structured input is where the lead is: at 8M, reverse 3.0-3.5x, nearly sorted
5.0-7.0x, concatenated halves 1.9-2.4x, rotated 1.7-2.6x, zigzag 2.1-2.8x, 256
distinct values 2.0-2.7x.  Uniform random is the one family that loses, and it
loses at every size for every key width except 8M int64/double, where the
second core pays for the extra passes.

**Read these as ranges, not as two decimal places.**  The same cell moves 20-60%
between runs on this box, so a single run cannot tell two builds apart.  Compare
builds by running both binaries back to back, several passes, and taking the
median of each column -- which is how the parallel narrow-range counting sort
was checked before it was committed (three paired passes; the random medians
differed by +11%, -14% and +4%, i.e. nothing, while lowcard16 and mod8 improved
1.2-2.3x).
