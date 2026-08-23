# FYX-SORT Ultimate — 设计思路（第一步：梳理）

本文件是实现 `fyx_sort.hpp` 之前的设计梳理，同时也记录了对原始需求中
**物理上无法实现的部分**的诚实说明。请先读第 0 节。

---

## 0. 关于验收标准的诚实说明（重要）

需求里有几条指标，不是"努力一下就能做到"，而是**违反内存带宽物理上限**的。
我不会假装做到，也不会用注释吹牛，下面给出数量级论证。

### 0.1 「单线程 1 亿 int32 ≥ 8 GB/s」不可能

- 1 亿 int32 = 400 MB。8 GB/s ⇒ **50 ms** 完成。
- 任何基于比较的排序在这个规模下都不可能（n log n ≈ 2.66×10⁹ 次比较，
  50 ms 意味着单核每秒 530 亿次比较）。
- 最快的路线是 LSD 基数排序：32 位键 + 8 位数字 = 4 趟，每趟一次顺序读 +
  一次随机散射写，来回 = 4 × (400 MB 读 + 400 MB 写) = **3.2 GB 内存流量**，
  外加直方图那一趟的 400 MB 读 ⇒ 约 3.6 GB。
- 单核可用内存带宽在主流服务器/桌面上是 **8–20 GB/s**（单核受限于
  line-fill buffer 数量，不是整机带宽）。取乐观的 20 GB/s，下限就是 180 ms，
  即「输入字节 / 时间」≈ **2.2 GB/s**。实测通常 0.8–1.5 GB/s。
- 结论：单线程上限大约是 **1.5 GB/s 量级**，8 GB/s 需要 5–10× 的带宽提升，
  换 CPU 也解决不了（HBM 机器例外，但那不是"任何环境"）。

### 0.2 「8 核 1 亿 int32 ≥ 30 GB/s」不可能

- 30 GB/s ⇒ 400 MB 在 **13 ms** 内排完 ⇒ 需要 3.6 GB / 13 ms ≈
  **270 GB/s** 的持续内存带宽。8 核桌面平台（双通道 DDR5）是 60–90 GB/s，
  高端服务器 8 通道 DDR5 也就 300 GB/s 峰值、实际 200 GB/s 左右，
  而且那是整机 64+ 核共享的。
- 现实目标：多线程基数排序能跑到接近内存带宽的 60–75%，
  在 8 核桌面上大约 **4–8 GB/s**（按输入字节计）。

### 0.3 「在任何设备上碾压世界上任何算法」无法保证

- ips4o / vqsort / x86-simd-sort 都是多年调优的成熟实现。本库的目标是
  **在同一数量级内竞争，并在若干场景（数值类型、已排序/低基数数据、
  小数组）上取胜**，而不是宣称通吃。任何声称"绝对第一"的库都在说谎，
  因为最优路径依赖于键分布、缓存大小、内存带宽、页大小和 NUMA 拓扑。

### 0.4 「A100 上 1 亿 int32 含传输 ≤ 10 ms」不可能

- 1 亿 int32 = 400 MB，PCIe 4.0 x16 实测 ~25 GB/s ⇒ **单向传输就要 16 ms**，
  往返 32 ms。加上排序本身（A100 上 CUB radix ≈ 3–4 ms），
  总时间下限约 **35 ms**。只有数据本来就在显存里（不含传输）才可能到 4 ms。
- 本库的 GPU 层用 pinned memory + 分段流水，把往返尽量与计算重叠，
  实测目标 ≈ 传输时间 + 20%。

### 0.5 「约 50000 行」

- 50000 行单文件里，真正有信息量的排序代码大概只能占 5–8k 行，其余必然是
  展开的模板实例、复制粘贴的网络或注释填充。膨胀的代码 = 编译慢（单个 TU
  几十秒）+ I-cache 压力 + 更多 bug 藏身处。
- 本实现约 **7k 行密集有效代码**，覆盖需求里列出的**全部**功能点，
  每一行都编译、都被测试跑到。如果确实需要行数，我可以再生成
  展开版（模板全实例化 + 逐算法注释），但那只会更慢更难维护。

**下面是实际实现的设计。所有性能数字最后都会在本机实测并写进 BENCHMARKS.md，
不写没跑过的数。**

---

## 1. 总体架构

```
fyx_sort.hpp
├── §1  配置宏 / 平台 / 编译器检测      (FYX_ENABLE_GPU, FYX_ENABLE_PARALLEL, FYX_DISABLE_AVX512 …)
├── §2  可移植原语                      likely/prefetch/popcount/ctz/aligned_alloc
├── §3  运行时 CPU 特征检测             CPUID + XGETBV，缓存为单例
├── §4  内存                            线程局部 ScratchBuffer（一次分配，重复使用）
├── §5  类型萃取                        连续容器检测 / 基数键编码 traits
├── §6  标量内核                        插入排序、堆排序、排序网络、中位数选取
├── §7  SIMD 排序网络                   AVX-512 / AVX2 / SSE4.2 / NEON 双调网络，n ≤ 64
├── §8  pdqsort                         无分支块分区 + 模式击破 + 堆排序兜底
├── §9  基数排序                        LSD 8 位；SIMD 直方图（vpconflictd+vpopcntd）；
│                                       SWWC 散射；AVX-512 scatter 路径；并行版
├── §10 工作窃取线程池                   Chase-Lev deque（正确的 seq_cst 栅栏 + CAS）
├── §11 并行采样排序                     ips4o 风格：采样 → 无分支分类 → 块置换 → 递归
├── §12 稳定排序                        并行归并 + 基数（LSD 天然稳定）
├── §13 partial_sort / nth_element      内省选择（quickselect + median-of-medians）
├── §14 自适应调度器                     大小/类型/分布/特征 → 路径选择
├── §15 公共 API + 容器适配 + C ABI
└── §16 GPU 层（#ifdef FYX_ENABLE_GPU） 动态加载 CUDA/HIP/OpenCL + NVRTC 运行时编译
```

## 2. 关键设计决策

### 2.1 数值类型：LSD 基数排序

- 8 位数字 × 4 趟（32 位）/ 8 趟（64 位）。
- **一趟扫描算出全部趟的直方图**（4 或 8 个 256 项计数器同时累加），
  这样只读一次数据。
- **跳过退化趟**：若某趟所有元素落在同一个桶，跳过；已排序/低基数数据
  因此近乎免费。
- **奇偶趟处理**：统计有效趟数，若为奇数则最后一趟直接写回原数组
  （通过交换 src/dst 指针决定，不做多余 memcpy）。
- **键编码**：
  - `uint*`：恒等
  - `int*`：翻转符号位 `x ^ (1<<(bits-1))`
  - `float/double`：`u ^ ((-(u >> (bits-1))) | signbit)` —— 负数全翻转，
    正数只翻符号位。这使 −NaN < −inf < … < −0 < +0 < … < +inf < +NaN，
    完全序，无 UB。
- **散射**：软件写合并缓冲（SWWC），每桶 64 字节（一个 cache line）缓冲，
  满则用 `_mm512_stream_si512` / `_mm256_stream_si256` 非临时写出，
  避免 RFO（read-for-ownership）浪费一半带宽。这是基数排序最重要的单点优化。
- **SIMD 直方图**（需求硬性要求「真 SIMD」）：AVX-512 路径用
  `_mm512_i32gather_epi32` 取计数 → `_mm512_conflict_epi32` 检测同向量内重复
  → `_mm512_popcnt_epi32` 得到重复计数 → `+1+popcnt` → `_mm512_i32scatter_epi32`
  写回。scatter 按 lane 顺序写，最高 lane 最后写入，恰好等于正确的累计值。
  需要 AVX512CD + AVX512VPOPCNTDQ，运行时检测；否则回退 4 路交错标量直方图。
  **实话**：在多数 CPU 上 4 路标量直方图更快（scatter/gather 微码开销大），
  所以 SIMD 直方图默认只在 `FYX_FORCE_SIMD_HISTOGRAM` 或检测到它更快时启用；
  两条路径都实现、都测试。
- **预取**：散射循环里对 `src + PF_DIST` 做 T0 预取，对目标桶指针做 T1 预取。

### 2.2 小数组（n ≤ 64）：SIMD 双调网络

- 不用插入排序（需求明确禁止）。
- 结构：把 n 补齐到 16/32/48/64（32 位类型）或 8/16/…/64（64 位类型），
  空位填 `max`（升序）哨兵，跑**完整双调网络**：
  `for k in 2,4,…,N: for j in k/2,…,1: 对 i 与 i^j 做定向 compare-exchange`。
- 当 `j < lanes`：向量内交换，用 `_mm512_permutexvar_epi32` + `mask_blend(min,max)`，
  索引向量与掩码在编译期由 constexpr 生成，无查表。
- 当 `j >= lanes`：跨向量，直接对两个向量做 `min/max`，方向在整个向量上一致
  （因为此时 `k ≥ 2j ≥ 2·lanes`，`i & k` 在一个向量内恒定）。
- 同一套代码用模板参数化 lane 数 / 类型，覆盖 int32/uint32/float/int64/uint64/double，
  AVX-512（16 或 8 lane）、AVX2（8 或 4 lane）、SSE4.2（4 或 2 lane）、NEON（4 或 2 lane）。
- 正确性验证：对 n = 1…64 × 所有类型 × 随机/重复/极值，与 `std::sort` 逐元素比对。

### 2.3 通用类型：并行采样排序（ips4o 风格）

- 采样 `α·k·log k` 个元素（k = 桶数 = 256 或 64），`std::nth_element` 取分位点，
  去重后建**无分支隐式二叉搜索树**（Eytzinger 布局），分类时用
  `idx = 2*idx + (cmp ? 1 : 0)` 的 `log k` 次无分支迭代（`cmov`）。
- 分类阶段每线程一段，写入本地块缓冲（每桶一个 block，默认 1024 元素）；
  块满则原子地追加到全局桶队列 —— 这是 ips4o 的核心，保证原地。
- 块置换阶段：按桶边界做块级环形置换（每个块整体搬运，无逐元素交换）。
- 清理阶段处理边界块的残余元素。
- 递归：桶大小 > 阈值继续采样排序；否则 pdqsort。
- 并行：桶间用工作窃取池分发；桶内大于 `n/threads` 时再并行。
- 内存受限或分配失败时回退到**并行 pdqsort**（任务化左右子区间）。

### 2.4 工作窃取（Chase-Lev）

- `top`/`bottom` 为 `std::atomic<int64_t>`，环形数组 `std::atomic<Task*>`。
- `push`：relaxed 读 bottom、relaxed 读 top（owner），必要时增长数组，
  `store(release)` 任务，`store(relaxed)` bottom，再 `atomic_thread_fence(seq_cst)`。
- `pop`：先减 bottom，`fence(seq_cst)`，读 top；空/单元素分支用 CAS 与 steal 竞争。
- `steal`：读 top（acquire）、`fence(seq_cst)`、读 bottom（acquire），
  `compare_exchange_strong(top, top+1, seq_cst, relaxed)`。
- 旧数组不释放（epoch 简化：保存在 vector 里，池析构时统一释放），
  避免 ABA 与悬垂指针。
- 线程池惰性初始化（`std::call_once`），静态生命周期，析构时 join。

### 2.5 自适应调度

入口 O(1)~O(√n) 的探测：
1. `n` 与 `sizeof(T)`；
2. 是否默认比较器 + 算术类型 ⇒ 基数排序候选；
3. 抽样 64 个位置估计有序度（升序/降序/常量）；全序直接返回，全逆序直接 reverse；
4. 抽样估计键的动态范围 ⇒ 极窄范围走计数排序；
5. 线程数、L2/L3 大小、AVX-512 可用性；
6. GPU（仅 `FYX_ENABLE_GPU`）可用且 n ≥ 100k ⇒ CPU/GPU 按实测吞吐比切分 + 归并。

### 2.6 GPU 层（可选）

- 三个后端：CUDA Driver API + NVRTC / HIP / OpenCL，全部 `dlopen`/`LoadLibrary`
  动态加载，符号缺失即判定不可用，静默回退 CPU。
- 内核源码以 raw string 嵌入：4 位数字 LSD 基数排序（per-block 直方图 →
  全局前缀和 → 有序散射），通用类型走双调归并。
- pinned memory + 多流分段，传输与计算重叠。
- 严格包裹在 `#ifdef FYX_ENABLE_GPU`，默认编译**一个 GPU 头文件都不包含**，
  也不引用任何 GPU 符号（用 `-Wl,--no-undefined` 可验证）。

## 3. 测试与验收

- **正确性**：n ∈ {0,1,2,…,100} ∪ {10³,10⁴,10⁵,10⁶,10⁷}；分布 = 随机 / 已排序 /
  逆序 / 全等 / 少量唯一值 / 管风琴 / 锯齿 / 负数 / 极值 / NaN / ±0 /
  denormal；类型 = 8 种数值 + 字符串 + 大结构体 + 自定义比较器；
  与 `std::sort` / `std::stable_sort` 结果逐元素比对（稳定性用 (key,idx) 对验证）。
- **接口**：vector / deque / array / list / C 数组 / 裸指针 / 迭代器对 / C ABI。
- **性能**：与 `std::sort`、`std::stable_sort`、`__gnu_parallel::sort`（若可用）
  对比，输出 GB/s 与加速比，写入 BENCHMARKS.md（**实测**，不编造）。
- **编译洁净**：`-std=c++17 -O3 -Wall -Wextra -Werror -pedantic`，
  GCC 12 实测；MSVC/Clang/MinGW 用条件编译保证语法可移植（本机无法实测，
  会在 README 明确标注哪些平台经过实测）。
