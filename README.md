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
  - `FYX_DISABLE_PARALLEL`（纯单线程）已验证可编译。
  - `FYX_DISABLE_SIMD` 当前在头文件里有一个**既有的、与本次 API 工作无关的编译问题**
    （`cpu_pause` / `store_fence` 在关掉 SIMD 时缺少 intrinsic 头），尚未修复；
    默认 `-march=native` 配置不受影响。

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

`fyx::sort` 在编译期/运行期自动选择内核。入口处有统一的输入分布预检测层：
先采样 1024 个元素判断是否值得全量验证；当样本显示已排序、逆序、全等或部分有序时，
再用一次线性扫描同时验证 monotonicity、等价性、 distinct≤256 上限 与相邻逆序边计数。低基数样本只作为候选信号，
真正提交前仍由现有计数排序路径完整验证，避免 profile 本身给低基数场景额外增加一趟 O(n) 税。所有提前退出都必须由线性校验证明，
不会靠猜测返回；明显随机高基数样本会跳过完整 distinct 预检，直接进入 radix / MSD / sample 路径。

| 条件 | 选择的内核 |
|---|---|
| 数值类型 + 默认 `<`/`>` + `n ≤ 64` | 分支无关 SIMD 双调网络（AVX-512 / AVX2 / SSE4.2 / NEON） |
| 已排序 / 全等 | 统一 profile 一次验证后直接返回 |
| 逆序（非 stable） | 统一 profile 一次验证后反转 |
| 部分有序（相邻逆序边 ≤ n/64，且规模合理） | pdqsort |
| 整数小值域 | 计数排序（O(n + range)） |
| 数值/浮点默认顺序低基数（≤256 radix keys） | radix-key 稀疏计数排序；大输入可并行计数/填充（O(n)，保留 `-0/+0`/NaN 总序语义） |
| 任意类型低基数（≤256 等价类） | 压缩计数排序，保留原始对象 payload；`stable_sort` 保持稳定 |
| 数值类型 + 默认 `<`/`>` | LSD 基数排序（8 位桶，SIMD/scalar 直方图，非临时散射写；大规模高熵 numeric 走 chunked parallel radix；2-worker/bandwidth-constrained 环境可对 64-bit/`float` random 先 MSD 分桶再桶内 radix/PDQ） |
| 默认顺序 `std::string` 大输入 | MSD 字节基数排序（随机字符串只读取区分前缀） |
| trivial struct + 比较器等价于整数 key 字段 | guarded comparator-key count/radix sort（采样确认语义 + 最终 `is_sorted` 校验） |
| 结构体 / 自定义比较器大输入 | 256 路 sample sort（ips4o 风格，块级桶重排 + 临时散射） |
| `stable_sort` 非低基数通用类型 | 自底向上归并排序（稳定） |
| `partial_sort` / `nth_element` | 堆 + 内省选择（quickselect） |

并行：当 `Options.parallel == On`（或 `Auto` 且问题规模够大且线程池可用）时，
高熵数值默认顺序在 3+ worker 上回到 chunked parallel radix（保留全 pass 快速规划、32-bit/value-buffer 优化，并用 32-bit local hist / 按 key 宽度自适应 chunk 数降低 64-bit recount/cache 压力；浮点 key-buffer 路径最后一趟直接 scatter+decode 回用户数组，少一次额外输出拷贝），避免 4H8G 上 MSD 分桶开销反噬；只有 2-worker/bandwidth-constrained 环境才启用 `float` top-byte / 64-bit top16 MSD-bucket hybrid；
低基数 radix-key 场景走并行 sparse counting；通用大输入走并行 sample sort（并行分类、散射和桶递归）；其它中等输入保留任务并行分治，
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
- `FYX_DISABLE_AVX512` / `FYX_DISABLE_AVX2` / `FYX_DISABLE_SSE42` / `FYX_DISABLE_NEON`
  （关掉某一档 ISA 的内核；`FYX_DISABLE_SIMD` 一次性关掉全部）
- `FYX_FORCE_SIMD_HISTOGRAM`（强制使用 AVX-512 冲突检测直方图）
- `FYX_NO_EXCEPTIONS`（分配失败时降级为就地算法而非抛异常）

---

## 测试

先由 `parts/` 生成头文件，再编译各测试：

```bash
./build.sh
for t in t_scalar t_net t_radix t_pdq t_pool t_deque_race t_api t_sample t_counting; do
  g++ -std=c++17 -O2 -march=native -pthread -Wall -Wextra -Werror test/$t.cpp -o /tmp/$t && /tmp/$t
done
```

`test/t_api.cpp` 覆盖：所有重载形态、`std::sort` / `std::stable_sort` 逐元素比对、
稳定排序的 (key,idx) 稳定性验证、`partial_sort` / `nth_element` 契约验证、
`-0`/`+0`/NaN 的浮点全序、以及 `extern "C"` ABI。
`test/t_counting.cpp` 覆盖低基数整数/字符串/结构体、稀疏 256 distinct、MSD 字符串边界、并行 sample sort 显式路径，以及武器七 profile 调度（已排序、逆序、全等、低基数、部分有序、高熵 comparator-key）。

---

## 性能

基础数值 benchmark 见 [`BENCHMARKS.md`](./BENCHMARKS.md)。IPS4o 对比 harness 与完整表格见 [`bench/README_ips4o_compare.md`](./bench/README_ips4o_compare.md)。

在本沙箱（2 vCPU Intel Xeon，GCC 12.2，`-O3 -march=native`）的 IPS4o sequential 与真实 oneTBB parallel 矩阵中，FYX 当前在已跟踪的 7 个分布上全部领先：

| case | FYX seq | IPS4o seq | FYX par | IPS4o par |
|---|---:|---:|---:|---:|
| i32 random, 5M | 0.0688s | 0.1299s | 0.0585s | 0.0687s |
| i32 low distinct 16, 5M | 0.0072s | 0.0184s | 0.0073s | 0.0108s |
| i64 sparse distinct 256, 3M | 0.0160s | 0.0197s | 0.0103s | 0.0114s |
| string random len16, 1M | 0.1540s | 0.1970s | 0.0941s | 0.1115s |
| string low distinct 64, 1M | 0.0212s | 0.0652s | 0.0209s | 0.0378s |
| struct key distinct 64, 1M | 0.0048s | 0.0052s | 0.0040s | 0.0050s |
| struct key random, 1M | 0.0138s | 0.0278s | 0.0127s | 0.0142s |

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
