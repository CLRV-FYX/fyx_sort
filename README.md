# FYX-SORT

单头文件、零依赖、高性能排序库。整个库就是 `fyx_sort.hpp`，`#include` 后即可使用，
所有代码在 `namespace fyx` 下。本库是「工程 / Ultimate」那条线：运行时 CPUID 特征检测、
Chase-Lev 工作窃取线程池、自适应算法调度、稳定排序接口、显式 `Options` 配置。

> 历史说明：早期文档里出现过「v9.1 / v9.2 两个版本」的说法，那是一种**过时的、与当前仓库不符**
> 的叙事。当前仓库只有这一个头文件（`fyx_sort.hpp`），它已经包含公开 API、稳定排序与自适应调度。
> 本文档以**当前仓库的实际代码**为准。

---

## 当前状态（诚实说明）

- **公开 API 已实现并测试通过**：`fyx::sort` / `fyx::stable_sort` / `fyx::partial_sort` /
  `fyx::nth_element`，以及指针+长度、迭代器对、容器、`Options`、`extern "C"` ABI 等全部重载形态。
- **验证环境**：Intel Xeon Ice Lake-SP，2 vCPU，g++ 12.2.0，
  `-O2 -march=native -pthread -Wall -Wextra -Werror`。
  既有内核测试（t_scalar / t_net / t_radix / t_pdq / t_pool / t_deque_race）全绿，
  公开 API、sample sort 与 counting/string/IPS4o 对比相关测试全绿。
- **性能数字只看 `BENCHMARKS.md`**（本机实测）。旧文档里那些「1 亿 int 2.5~4 秒」之类的
  营销数字**没有在本环境实测过**，请当作未经证实的参考，不要照搬到你的机器上。
- **我们能站得住脚的承诺**：在本机上，对测试过的分布，`fyx::sort` **不会比 `std::sort` 慢**，
  对数值类型则明显更快（见 `BENCHMARKS.md`）。这是可以复现、可以验证的。
- **已知限制**：
  - 多核 ≥30 GB/s、A100 上含传输 ≤10 ms 等目标**无法在本沙箱验证**（仅 2 个硬件线程、无 GPU）。
  - 编译配置矩阵已逐个验证可编译**并跑通测试**：默认、`FYX_DISABLE_PARALLEL`、
    `FYX_DISABLE_SIMD`（曾经是已知缺陷，现已修好：`_mm_pause` / `_mm_sfence` 不是向量内核，
    但住在 SSE 头里）、`-fno-exceptions` / `FYX_NO_EXCEPTIONS`、`FYX_DISABLE_AVX512`、
    `FYX_DISABLE_AVX2`、`FYX_ENABLE_FAST_PATHS=0`、`FYX_USE_PDQ_PARTITION=0`、
    `FYX_SAMPLE_SORT_V2=0`、`FYX_USE_STRING_VIEW=0`、`FYX_ENABLE_GPU=1`（含
    `FYX_GPU_COMPUTE=1`，那条 NVRTC 路径以前连编都编不过：raw string 用 `R"CUDA(` 开、
    却用 `)"` 收），以及 C++17/20 × `-O0`/`-O3`（不带 `-march`）。
    `FYX_DISABLE_SIMD` 与 `FYX_DISABLE_PARALLEL` 各自跑完整 API 套件（847 项）全绿。
    注意 `-fno-exceptions` 现在**不需要**同时定义 `FYX_NO_EXCEPTIONS`：GCC/Clang 在
    `-fno-exceptions` 下是不定义 `__cpp_exceptions`（而不是定义成 0），原来的特性检测抓不到。
  - 没有 MSVC / ARM / GPU 硬件可供本机验证，那几条路径只有代码与编译期开关，没有实测数字。

---

## 快速开始

```cpp
#include "fyx_sort.hpp"
#include <vector>
#include <algorithm>

std::vector<int> v = {5, 3, 8, 1, 9, 2};

fyx::sort(v);                        // 容器重载
fyx::sort(v.begin(), v.end());       // 迭代器对；vector/string 等连续迭代器会进入同一指针快路径
fyx::sort(v.data(), v.size());       // 指针 + 长度
fyx::sort(v, std::greater<int>());   // 自定义比较器（降序）

fyx::stable_sort(v);                 // 稳定排序（数值升序走基数，天然稳定）
fyx::partial_sort(v.begin(), v.begin() + 3, v.end());          // 前 3 个最小且有序
fyx::nth_element(v.begin(), v.begin() + v.size()/2, v.end());  // 中位数到位

// C ABI（C++ TU 内调用；从 C 调用需单独编译一个引用这些符号的 .cpp）
int32_t a[1024];
fyx_sort_int32(a, 1024);
```

编译（SIMD 内核自带 target 属性，运行时分发，因此 `-march=native` 不是必须的）：

```bash
g++ -std=c++17 -O3 -pthread your_program.cpp
# 想要榨干 CPU：
g++ -std=c++17 -O3 -march=native -pthread your_program.cpp
```

MSVC：`cl /EHsc /std:c++17 /O2 /arch:AVX2 your_program.cpp`

---

## 算法与自适应调度

`fyx::sort` 在编译期/运行期自动选择内核。入口处有可用 `FYX_ENABLE_FAST_PATHS` 关闭的快速分布层：
算术类型全等先用 shifted `memcmp` 证明，字符串大输入全等可并行验证；已排序用一次线性扫描证明后直接返回，逆序可在验证时同步交换以减少第二趟 reversal 成本；
adjacent-swap zigzag 先用 in-place pair repair 验证并直接交换相邻逆序对；bench_final 风格的“前半反转、后半升序” organ-pipe zigzag 会在 reverse 快速路径之前识别，证明后只反转前半段；局部 sawtooth/小位移 zigzag 先用样本位移预算保护的 bounded insertion repair；half-organ/bitonic zigzag 会先识别前后两个单调 run，连续数值域直接算术填充，固定半幅字符串 organ 形态用半缓冲交错重排，其他对象走线性两 run merge；更一般的交错 zigzag 仍会被识别为偶/奇两个单调 run（四种方向组合），当两个 run 只需串接时改用半缓冲 deinterleave，其他情况再线性 merge。随后统一 profile 层再采样 1024 个元素判断是否值得全量验证；
当样本显示已排序、逆序、全等或部分有序时，再用一次线性扫描同时验证 monotonicity、等价性、 distinct≤256 上限 与相邻逆序边计数。低基数样本只作为候选信号，
真正提交前仍由现有计数排序路径完整验证，避免 profile 本身给低基数场景额外增加一趟 O(n) 税。所有提前退出都必须由线性校验证明，
不会靠猜测返回；明显随机高基数样本会跳过完整 distinct 预检，直接进入 radix / MSD / sample 路径。

| 条件 | 选择的内核 |
|---|---|
| 数值类型 + 默认 `<`/`>` + `n ≤ 64` | 分支无关 SIMD 双调网络（AVX-512 / AVX2 / SSE4.2 / NEON） |
| 已排序 / 全等 | 快速分布层或统一 profile 一次验证后直接返回；算术全等优先 shifted `memcmp` |
| 逆序（非 stable） | 快速分布层可验证时同步交换；否则统一 profile 一次验证后反转 |
| adjacent-swap / local sawtooth zigzag | adjacent pair 先证明后原地交换；更小块 sawtooth 用样本保护的 bounded insertion repair，随机/低基数样本会快速拒绝 |
| 前半反转 organ-pipe zigzag | 验证前半非增、后半非降且边界有序后，只反转前半段；避免被 reverse 探测误送 pdqsort |
| half-organ / bitonic zigzag | 数值连续域验证两半 ±2 步长后直接填充；字符串半幅 organ 可半缓冲交错重排；其他类型验证单转折后线性 merge 两个单调 run |
| 交错 zigzag 两 run | 验证偶/奇位置各自单调；可串接时半缓冲 deinterleave，重叠时线性 merge |
| 部分有序（相邻逆序边 ≤ n/64，且规模合理） | adjacent/local repair 先处理；数值默认顺序的长距离 nearlysorted 先检测连续整数/整数值浮点 permutation 并直接填充，否则进入 radix，避免 pdq 比较瓶颈；非数值长距离扰动用 dirty-patch merge，失败才退到 pdqsort；浮点默认顺序使用 radix-key comparator 以保留 NaN / `-0/+0` 总序 |
| 整数小值域 | 计数排序（O(n + range)）；大输入先用 sample-gated 并行 dense-range count/fill 加速 256-way lowcard |
| 数值/浮点默认顺序低基数（≤256 radix keys） | radix-key 稀疏计数排序；大输入可并行计数/填充，浮点/32/64-bit 稀疏键先尝试 rank16 direct-map 计数，dense integer range 与浮点 compact prefix direct-map 仍优先处理小值域（O(n)，保留 `-0/+0`/NaN 总序语义） |
| 任意类型低基数（≤256 等价类） | 压缩计数排序，保留原始对象 payload；`stable_sort` 保持稳定 |
| 数值类型 + 默认 `<`/`>` + 高基数 + 有 AVX-512（`int32/uint32/float/double`，n ≥ 16384） | **AVX-512 向量快排**：`vpcompressd`/`vcompressps` 原地分区（每轮 4 个向量、读哪一侧用 cmov 而不是分支），叶子是寄存器内 Batcher 双调网络（16 个向量：4 字节类型 256 个元素、8 字节 128 个），枢轴取两向量跨步样本的中位数；并行时前几层分区后把两侧交给线程池。浮点先做一次只读扫描，发现 NaN 或 `-0` 就让位给 radix（硬件比较复现不了那个总序）。64 位整数仍走 high-prefix radix：它两趟就能排完 24–26 位前缀，比快排的层数便宜 |
| 数值类型 + 默认 `<`/`>` | LSD 基数排序（8 位桶，SIMD/scalar 直方图，非临时散射写；大规模高熵 32-bit integer random 先走 10/11/11 三趟 wide-key radix，**64-bit random 走 AVX-512 向量快排**（compress-store 分区 + 网络叶子；在两台实测机器上对 high-prefix radix 是 2–9 倍，含其 40-bit 宽分布 tie 修复的 9 倍病态案例，见 tools/dev/NOTES.md），`double` 在 1M 级别排序 encoded top36、较大输入保留 top39，并在最后一趟解码；其它 numeric 回退 chunked parallel radix；2-worker/bandwidth-constrained 环境仍保留 MSD-bucket hybrid fallback） || 默认顺序 `std::string` 大输入 | MSD 字节基数排序（随机字符串只读取区分前缀）；低基数字符串在 comparator 模式下也可走 unordered value-count/fill，大输入会并行计数/填充且只对 distinct key 排序 |
| 自定义比较器但采样等价于自然升/降序的数值或 `std::string` | guarded radix/count/MSD recovery（采样确认 + 最终 `is_sorted(comp)` 校验；高熵 64-bit/`double` comparator 串行路径也可用自适应 high-prefix radix；失败则继续比较排序） |
| trivial struct + 比较器等价于整数 key 字段 | guarded comparator-key count/radix sort（采样确认语义 + 最终 `is_sorted` 校验） |
| 结构体 / 自定义比较器大输入 | 256 路 sample sort（cheap/trivial payload 使用 unrolled Eytzinger 分类；并行 arithmetic comparator fallback 顶层用 64 路 partition + 256 路采样预算以减少 random 数值分类比较；`std::string` fallback 保持 looped classifier + 128K handoff + block scatter；非字符串串行路径单趟 prefix scatter，并行路径分 chunk 计数/散射；低 distinct 算术样本先走 rank16/sparse exact value counting，float/double 256-way comparator 低基数也优先尝试该 exact counter） |
| `stable_sort` 非低基数通用类型 | 自底向上归并排序（稳定） |
| `partial_sort` / `nth_element` | 堆 + 内省选择（quickselect） |

并行：当 `Options.parallel == On`（或 `Auto` 且问题规模够大且线程池可用）时，`Auto` 默认从约 1,000,000 元素开始启用线程池，并同时检查每个线程至少约 128 KiB 数据；可用环境变量 `FYX_MIN_PARALLEL_SIZE` 覆写最小并行规模。
高熵数值默认顺序会先尝试 32-bit integer 10/11/11 三趟 wide-key radix（减少一轮全数组 count/scatter，1M 级别用 1 chunk/worker 降低调度/WCB 开销，并在 ≤2M 元素时关闭 NT store 以减少短数组写合并成本；较大输入保留 2 chunks/worker 的粗粒度调度）**64-bit 高熵随机已改配 AVX-512 向量快排**（`vqsort_preferred` 的 int64/uint64 翻转有跨两台机器的配对 A/B 支撑：旧机器 radix 赢 7%，本机 vqsort 赢 2–9 倍——取最坏情况更小的那个），用更少全数组 pass 攻击 random 短板；不适用时回到 chunked parallel radix（保留全 pass 快速规划、32-bit/value-buffer 优化，并用 32-bit local hist / 按 key 宽度自适应 chunk 数降低 64-bit recount/cache 压力；64-bit 整数 value-buffer radix 保持 4H2G 实测更稳的 3 chunks/worker；`double` key-buffer 路径最后一趟直接 scatter+decode 回用户数组），2-worker/bandwidth-constrained 环境仍保留 `float` top-byte / 64-bit top16 MSD-bucket hybrid fallback；
低基数 radix-key 场景走并行 sparse counting；整数 64..65536 小值域会优先走 sample-gated 并行 dense-range count/fill，浮点/32/64-bit 稀疏低基数会先尝试 collision-free rank16 direct-map，浮点小值域也保留 compact prefix direct-map，以避免逐元素 hash probe；通用大输入走并行 sample sort（并行分类、散射和桶递归）；其它中等输入保留任务并行分治，
归并阶段优先使用 scratch-buffered 并行分块归并，分配失败或不适用时回退到 `std::inplace_merge`。默认 `Auto`。

---

## Options

```cpp
fyx::Options o;
o.parallel = fyx::Tri::Auto;   // Auto | Off | On
o.threads  = 0;               //  advisory，0 = 使用线程池默认大小
fyx::sort(v, o);              // 或 fyx::sort(v.begin(), v.end(), o)
```

---

## 编译期开关（均可选，默认「直接 include 就能用」）

- `FYX_ENABLE_PARALLEL`（默认开；定义 `FYX_DISABLE_PARALLEL` 得到无 `<thread>` 依赖的纯单线程构建）
- `FYX_ENABLE_FAST_PATHS`（默认开；关闭后禁用入口 sorted/all-equal/reverse/zigzag 快速分布层）
- `FYX_USE_PDQ_PARTITION`（默认开；控制部分有序/zigzag 的 pdq/两 run 处理）
- `FYX_USE_STRING_VIEW`（默认开；字符串全等验证使用 `data()+size()`/`char_traits::compare`，避免构造 view/拷贝）
- `FYX_SAMPLE_SORT_V2`（默认开；保留 sample-sort v2 调整的编译期开关）
- 运行期环境变量：`FYX_MIN_PARALLEL_SIZE=<元素数>` 可覆写 Auto 并行最小规模。
- `FYX_DISABLE_AVX512` / `FYX_DISABLE_AVX2` / `FYX_DISABLE_SSE42` / `FYX_DISABLE_NEON`
  （关掉某一档 ISA 的内核；`FYX_DISABLE_SIMD` 一次性关掉全部）
- `FYX_FORCE_SIMD_HISTOGRAM`（强制使用 AVX-512 冲突检测直方图）
- `FYX_NO_EXCEPTIONS`（分配失败时降级为就地算法而非抛异常）

---

## 测试

先由 `parts/` 生成头文件，再编译各测试：

```bash
./build.sh
for t in t_scalar t_net t_radix t_pdq t_pool t_deque_race t_api t_sample t_counting t_adaptive t_vsort; do
  g++ -std=c++17 -O2 -march=native -pthread -Wall -Wextra -Werror test/$t.cpp -o /tmp/$t && /tmp/$t
done
```

`test/t_api.cpp` 覆盖：所有重载形态、`std::sort` / `std::stable_sort` 逐元素比对、
稳定排序的 (key,idx) 稳定性验证、`partial_sort` / `nth_element` 契约验证、
`-0`/`+0`/NaN 的浮点全序、以及 `extern "C"` ABI。
`test/t_vsort.cpp` 覆盖 AVX-512 向量快排：10 种形状 × 17 种规模 × 6 种类型，
内核直调（串行与并行驱动）与公开入口（升序/降序/stable）逐元素对比 `std::sort`；
形状里包含快排会退化的那些（90% 是最小值、两个值、全等、周期）；
浮点另外验证「有 NaN 或 `-0` 时内核必须让位」以及最终仍是 radix 总序。
`test/t_counting.cpp` 覆盖低基数整数/字符串/结构体、稀疏 256 distinct、MSD 字符串边界、并行 sample sort 显式路径，以及快速/profile 调度（已排序、逆序、全等、低基数、部分有序、floating repair、front-reversed-organ/adjacent-swap/local-sawtooth/interleaved/half-organ zigzag、长距离 nearlysorted 数值 permutation/radix、高熵 comparator-key）；还有大 n（100 万/200 万/400 万）的**并行有序性证明**：串行与并行两条路在 8 种形状 × 升序/降序上必须给出完全相同的裁决与调度决定（含专门构造来骗过 4096 点抽样的形状），并端到端验证 200 万以上的升序/降序/逆序/全等排序结果与 `std::sort` 一致；还有同样规模的**池辅助证明**（200 万以下、仅借池做验证）等价性块——含「全等前缀 + 严格递减尾部」这类专门考验前缀分类与块缝合并的对抗形状，以及 `try_order_exit_adaptive` 开/关池两条路由必须给出相同裁决。
---

## 性能

完整表格、每一格的数字和生成脚本见 [`BENCHMARKS.md`](./BENCHMARKS.md)（由 `tools/dev/mkbench_md.py` 从 `build/vqsort_<n>.txt` 生成，没有手写数字）。

**矩阵里有六个对手**：`std::sort`、`pdqsort`、**`vqsort`（Google Highway 1.4.0）**、
**`IPS4o` 串行与并行（oneTBB）**、**`x86-simd-sort`（Intel 的 AVX-512 排序库）**。
`bash tools/dev/vqsort.sh <n>` 会把这一整套拉取、构建、跑完并写入 `build/vqsort_<n>.txt`
（缺 oneTBB 时会构建「IPS4o 仅串行」的缩减矩阵，而不是伪造并行列）。
每格是「六者中最快的那个 / 并行 `fyx::sort`」，1.00x 打平，小于 1 是输。
覆盖 4 种类型 × 14 种分布，`string` 只在 100 万规模跑（vqsort / x86-simd-sort 不吃字符串）。

| 规模 | 胜 | 负 | 说明 |
|---|---:|---:|---|
| 100 万 | 33 | 9 | 输的格子：周期为 8（0.66x/0.68x/0.99x）、16 个不同值与全等（0.91x–0.99x）、块交换 double（0.82x） |
| 800 万 | 36 | 6 | 输的格子：块交换 double（0.71x）、块交换 int32（0.92x）、周期为 8 int32（0.93x）、锯齿波 int32（0.92x）、全等 int64 与锯齿 double（0.97x–0.98x） |

**随机均匀键这一族已经翻过来了**：1M 随机 int32 从 0.0048 s 降到 0.0025 s（vqsort 0.0033 s，
1.32x），随机 double 0.0068 → 0.0054 s（1.22x），随机 int64 1.47x；800 万上三种类型分别是
1.03x / 1.25x / 1.40x。做法是给高基数数值输入加了一条 **AVX-512 向量快排**（见上表），
LSD radix 的「int32 至少三趟、每趟约 3.1 ns/elem」下界不再是这条路的天花板。
结构化数据（已排序、逆序、近似有序、拼接、旋转、低基数、字符串）仍然全线领先，多数在 2–7 倍。
剩下的输的格子集中在「值很少但周期性」的输入和块交换 double 上。
完整表格和每一个输的格子的数字见 [`BENCHMARKS.md`](./BENCHMARKS.md)。
**本轮把最后一批 0.9x 格子清零的两项通用改动**：

1. **向量化有序性证明**（`radix_key_monotone_scan`）：琐碎输入（已排序/逆序）的
   排序就是一次有序性扫描，浮点在基数全序下每元素要重推一次编码键——标量循环
   每元素约 6 个操作。换成 AVX-512：每 8/16 个元素一次加载 + 三条编码指令 + 一次
   `alignr` 移位自比较 + 一次掩码测试，序违例即早退。键是位精确的，浮点 NaN/-0
   总序不变；判定与标量探测器等价（t_counting 有等价性测试）。两条证明路径
   （<2M 池辅助、≥2M 全并行）共用。1M double 已排序 1.57→1.93x，8M 从
   0.94x 输 IPS4o 并行翻成 1.30x 赢，double 逆序 8M 1.78→2.17x。
2. **向量化全等扫掠**（`range_all_equal_first_vec`）：全等输入只需"每元素与
   p[0] 比较"，不再需要逐 chunk 三向分类。每 64 字节一次比较 + 全掩码检查，
   -0/+0 仍按编码键区分（与探测器判定一致）；有异议元素回落原分类路径。
   1M int64 全等 0.95x → 3.17x、double 1.07x → 3.10x、int32 0.91x → 1.22x；
   8M int64/double 全等 1.3x → 3.8x。

**低基数计数核上轮重写**（细分四步，全部同块 A/B 验证）：值域远宽于键数的形状
（矩阵 mod8 = 8 个键摊在 28672 个槽上）改走 rank 计数，核开销从 O(range×chunks)
降为 O(n+d)；d≤16 时计数趟换成 AVX-512 比较式直方图（每键每向量块一次比较 +
一次掩码加，去掉"哈希→查表→验证"的依赖加载链，未采样键表现为总数短缺而拒绝，
正确性不靠逐元素验证）；fill 按输出区间切分（旧按 rank 切分在 d=8 时退化为单任务，
800 万字节全落一个核）；int32 mod8 的 fill 保留按 rank 粒度（输出区间切分实测慢 16%）。
另外修掉一个池的潜伏竞态：外部线程此前全部假冒 worker 0 推弹同一条 Chase-Lev deque
（单所有者协议被破坏，8 线程并发排序会偶发挂死），现在第一个外部线程认领 worker 0，
其余走互斥 FIFO 由空闲 worker 抽干——修复后 30 次压测零挂死，TSan 无警告。

分对手看（并行 `fyx::sort` 对该对手的比值，最差 / 最好）：

| 对手 | 1M 胜/负 | 1M 最差 | 1M 最好 | 8M 胜/负 | 8M 最差 | 8M 最好 |
|------|---------:|--------:|--------:|---------:|--------:|--------:|
| std::sort | 56/0 | 1.37x | 131x | 42/0 | 1.65x | 109x |
| pdqsort | 56/0 | 1.02x | 13x | 42/0 | 1.18x | 15x |
| vqsort | 42/0 | 1.01x | 33x | 42/0 | 1.30x | 26x |
| IPS4o 串行 | 56/0 | 1.57x | 51x | 42/0 | 1.54x | 32x |
| IPS4o 并行 | 56/0 | 1.03x | 54x | 42/0 | 1.07x | 19x |
| x86-simd-sort | 42/0 | 1.03x | 37x | 42/0 | 1.25x | 25x |

**这一轮换来的最大改善来自新对手**：矩阵里加入 IPS4o 并行后立刻暴露出一类真实损失——
**已排序的 800 万输入上，IPS4o 的「检查是否已排好」是并行的，我们是单线程扫描，于是
它的整体调用比我们快 1.6–2 倍**（int32 0.60x、int64 0.48x、double 0.65x）。

8M 的修法是：抽样门先用 4096 个等距探针定方向（两个方向都不成立就在微秒级放弃，随机/旋转/
块交换这些形状因此几乎不付代价）；样本给出方向且见到严格有序的一对之后，「单调或不单调」
只剩下样本那个方向可能是真，于是每个 worker 只验证那一个方向——每元素一次比较，每个
chunk 额外验证跨缝的那一对，因此扫完整段就是完整证明。改成逐 chunk 分类（每元素两次
比较）时 8M int64 要 0.0054 s，改成单向验证后是 0.0034 s；浮点改成缓存上一个元素的
*编码键*、每元素只编码一次（两次编码实测慢 3 倍）。

**200 万以下的已排序是同一缺陷的第二种形态**：排序本体走串行，检查却连线程池都不碰。
修法是「池辅助证明」——排序保持串行，只把 *证明扫描* 借给线程池。这里有个实测教训：
直接复用 8M 的 4096 点 *跨步* 抽样门反而更慢，刚被调用方写入过的 1–8 MB 数组上，4096 个
跨步探针是约 0.3 ms 的纯冷未命中，比它把守的扫描还贵。所以辅助路径改用 *顺序前缀*：
先用串行检测器分类前 4096 个元素（顺序读、预取友好），乱序输入——随机、近似有序、旋转——
在这一步微秒级放弃，**永远不会唤醒线程池**；前缀单调才把剩余部分的验证按单向、每元素
一次比较铺到池上，前缀加各 chunk（含跨缝对）并起来恰好是整段，仍是完整证明。
1M 已排序现在是 int32 1.18x、int64 2.31x、double 2.13x、string 1.03x；8M 是
int32 1.08x、int64 1.07x、double 1.29x（2026-09-16 净窗口中位）。两条证明路与串行检测器的
等价性在 `test/t_counting.cpp` 里有专门的对抗形状测试；上一轮把证明扫描本身
向量化（每 8/16 个元素一次加载 + `alignr` 移位自比较，见下文），当时 1M double
已排序从 1.57x 提到 2.01x，8M double 从 0.94x 边缘翻成 1.34x。

**逆序反转与浮点小值域的两处通用改造**：① 反转 800 万 double 是 128MB 的纯带宽活，
原来是一条串行交换趟——现在证明趟（本来就并行）之后，镜像交换按分段铺到线程池
（每对 (i, n-1-i) 相互独立，任何多核机器都随真实内存带宽扩展；2 核带宽封顶时持平，
核越多收益越大）。② 密集窗口计数原本只收整数，但浮点的编码键就是保序整数——
double 的「编码值落在窄窗口」列（评分、定标价格、`i % 8` 这类）现在与整数走同一个
O(n+窗口) 计数器，不再走哈希探测的稀疏路径。现在逆序 1M 是 int32 1.90x、int64 1.49x、
double 1.90x，8M 是 1.85x / 1.87x / 2.18x；mod8 全家族 1.40–2.02x。

**随机均匀键这一族仍然领先**：1M 随机 int32 1.53x、double 1.56x、int64 1.47x；
8M 上 int32 1.28x / int64 1.63x / double 1.52x（对 vqsort / x86-simd-sort 本尊）。
做法是给高基数数值输入加了一条 **AVX-512 向量快排**（compress-store 分区 + 寄存器内双调
网络叶子），LSD radix 的「int32 至少三趟、每趟约 3.1 ns/elem」下界不再是这条路的天花板。
**纯比较模式**：传一个自定义比较器（库检测到非原生序时自动放弃基数/计数/向量快排，
只剩单调扫描 + 样本排序 + pdq 的纯比较栈）后，对 IPS4o **串行**全形状全胜
（随机 1.6–4.7x、已排序 1.4–1.8x、逆序 1.3–1.4x）；对 IPS4o **并行** 800 万全胜
（1.04–2.38x）、逆序大胜（10–24x，其并行排序在逆序输入上有病态），100 万为平手级
（int64 0.96–0.99x、double 0.98–1.06x，三独立进程交错测量；早前记过的 0.84x
「输格」三次独立复测均未出现，判定为单会话噪声并撤回）。测量记录见 `tools/dev/NOTES.md`
（含两次测量伪差的发现与更正过程）。

结构化数据（已排序、逆序、近似有序、拼接、旋转、锯齿、低基数、字符串）在六个对手面前
依然全线领先，多数在 2–25 倍。

**没有输格了**。两个规模、四种类型、十四种分布，对六个对手的 98 个格子全部高于
1.00x——最弱的几格是 int32 全等 1.01x（vqsort）、int64 远距离交换 1.02x（pdqsort）、
int64 8M 已排序 1.07x 与 string 已排序 1.03x（IPS4o 并行；2026-09-16 净窗口
3 进程中位复测）。把最后一批 0.9x 格子清零靠的是两项本轮落地的向量化改造：**有序性证明扫描**（浮点每元素
要重推编码键，标量循环 ~6 ops/元素；现在每 8/16 元素一次加载 + 三条编码指令 +
`alignr` 移位自比较 + 一次掩码测试，违例早退，位精确键保持 NaN/-0 总序）和
**全等扫掠**（全等判定 = 每元素与 p[0] 比较，免三向分类免接缝簿记；全掩码按通道宽
计数，-0/+0 仍按编码键区分）。完整表格与每一格的数字见 [`BENCHMARKS.md`](./BENCHMARKS.md)；
**尚未清零的战线**见下文「剩余的优化前沿」。

### 剩余的优化前沿（按收益排序，全部是诚实测量出来的）

1. **串行随机算术列落后 SIMD 排序专精 10–16%**。并行列碾压的前提是池；把线程拿掉
   （单核机器、调用方 `threads=1`、嵌套在别人池里的场景），1M 随机 int32/int64/double
   对 vqsort 本尊为 0.86x / 0.85x / 0.89x，8M 同量级（0.86x / 0.86x / 0.86x；
   2026-09-16 七轮同块交错中位）。本轮已落地「lean+max 分区」：热循环只跟踪区间最大值
   （每向量一条向量指令，旧版 min+max 是两条；最小值与 `split==0` 检测冗余），保留两个
   免费早退（整段同值直接返回、右侧全 pivot 丢弃不递归）——int64/double 各捞回 2–3 个
   点，int32 持平。dispatch 轨迹证实**三种类型（含 int32）的串行随机全部路由 AVX-512
   向量快排**（早先「int32 串行走 wide-key radix」的说法被证伪并撤回）。剩余差距的
   候选：hwy 的 BaseCase 排序网络、PartitionRightmost 的余数预处理、GatherSample 枢轴。
   下一轮从这里开始。
2. **纯比较模式的 1M 并行平手**（int64 0.96–0.99x、double 0.98–1.06x，对 IPS4o 并行）。
   自定义比较器不可编码成键，向量化的证明与计数都不适用，样本排序的比较深度是
   主要成本——大概率是诚实边界，除非给比较器式样本排序做自适应过采样。
3. **string 已排序 1.03x、string mod8 1.18x**（对 IPS4o 并行）。字符串验证是比较器
   短路扫描（memcmp 主导），双方都是带宽受限，空间很小。
4. **近平手格**：int32 全等 1.01x（vqsort）、int64 远距离交换 1.02x（两段拼接修复的
   patch 合并）、int64 8M 已排序 1.07x（采样门 + 池唤醒的 ~10µs 固定开销；1M 已是
   2.31x）。都是微优化量级。
5. **泛化性边界（本机测不了的部分）**：本机 2 核，IPS4o 并行也只拿到 2 线程——4 核以上
   机器上对手会更快，我们的池也随核数扩展，相对位置待第三方复测；NEON 机器上本轮的
   AVX-512 改造（向量计数、向量化证明、比较式全等扫掠）回落标量路径，**行为正确但
   加速比不成立**，NEON 等价加速是独立的后续工作。

几点关于这张表的诚实说明：

- **本机只有 2 个硬件线程**，所以 IPS4o 并行也只拿到 2 个线程；在更多核的机器上它会比
  本表更快，本表对它的相对位置是偏乐观的。反过来，`vqsort` / `x86-simd-sort` 是单线程库，
  在 2 线程机器上吃不到额外并行。
- **同一格重复跑会有 20–60% 的摆动**（见 `tools/dev/NOTES.md` 的测量记录），所以比值请
  当作区间看，0.96x–1.05x 之间基本是噪声；要比较两个版本请做背靠背交替测量。
- **`bench_matrix --filter=` 跑出来的数据与全量跑不是同一份**：所有数据集来自同一个按矩阵
  顺序推进的随机流，过滤会改变流前缀。只有已排序/逆序/全等/周期为 8 这几个不依赖随机流的
  形状可以用过滤跑复核。
- 早期那份 7 个分布的 IPS4o 对比 harness 仍在 [`bench/bench_ips4o_compare.md`](./bench/README_ips4o_compare.md)，
  数字是旧版本的，以本表（`tools/dev/vqsort.sh` 可复现）为准。
---

## 许可证

本库采用**署名-非商业性使用**许可。

使用本库（无论整体还是部分）必须在以下至少一个位置注明作者和 GitHub 来源：

- 源文件头部注释
- 项目 README 或文档
- 最终产品的“关于”或“致谢”页面

注明格式：

```
排序算法基于 FYX-SORT (https://github.com/你的用户名/fyx-sort)
作者：付yanxin (FYX)
```

商业使用（闭源商业软件、SaaS、嵌入式产品等）需要单独联系作者获取授权。
