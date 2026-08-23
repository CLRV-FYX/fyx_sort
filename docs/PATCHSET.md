# FYX-SORT 补丁集：全平台 / 全环境 / 全类型强化方案

> 本文档是对 11 项审计条目的**逐条实测复核** + 完整补丁方案。
> 复核环境：Intel Xeon Ice Lake-SP（family 6 model 106）、2 vCPU、g++ 12.2.0、cgroup v2 容器内。

---

## 第 0 节：审计条目实测复核（先纠错，再动手）

动手前我把 11 条全部拿到代码和机器上验证了一遍。**7 条属实，2 条前提错误，2 条机制判断有误**；另外我发现了 **1 个审计清单里没有、但同样致命的 bug**。

| # | 审计结论 | 复核结果 | 证据 |
|---|---|---|---|
| 1 | MSVC 下 `FYX_TARGET_*` 为空 → Windows 报废 | ✅ **属实，最高优先级** | `parts/01_config.hpp:357-361` 非 GNU 分支全部展开为空 |
| 2 | 许可证是 CC BY-NC，商业不可用 | ❌ **前提错误** | `LICENSE` 第 1 行即 `Apache License Version 2.0`。仓库**从一开始就是 Apache-2.0**，无需变更 |
| 3 | AVX-512 无规模门槛 → 降频拖累中小数组 | ⚠️ **现象属实，机制判断错误** | 见下方实测表。真因是**填充浪费**，不是降频 |
| 4 | 非数值类型回退 pdqsort，无块采样 | ✅ 属实 | `grep sample_sort parts/` 无结果 —— 采样排序尚未实现，本就在路线图上 |
| 5 | `stable_sort` 仍是普通归并，慢于 ips4o 的旋转归并 | ❌ **前提错误（双重）** | (a) `stable_sort` **根本还不存在**，没有"仍是"；(b) **ips4o 不是稳定排序算法**，不存在可对标的"ips4o 旋转归并" |
| 6 | 并行阈值写死 32768，不感知 cgroup | ✅ **属实**，但建议的实现路径是错的 | `kParallelThreshold = 1u << 15` 硬编码；实测本机 `/sys/fs/cgroup/cpu.max` **不存在**（见 Module A） |
| 7 | Chase-Lev 的 `seq_cst` fence 在 ARM 上过重 | ⚠️ **理论成立，本沙箱无法验证，且不能简单删除** | `uname -m` = x86_64，无 aarch64 交叉编译器/qemu。fence 是算法正确性的一部分 |
| 8 | 小数组网络只支持 32/64 位 | ✅ 属实 | 8 个 Ops 策略全部是 `Key = uint32_t` 或 `uint64_t` |
| 9 | `BitonicVec` 重复 4 份 → I-Cache 压力大、编译慢 | ⚠️ **编译慢属实，I-Cache 压力不成立** | 编译实测见下；运行时只有一条 ISA 路径被执行，其余代码页从不进入 I-Cache |
| 10 | `ScratchLease` 嵌套时退化为 malloc | ✅ 属实 | `parts/04_memory.hpp:110`，`if (!a.in_use())` 失败即私有 malloc |
| 11 | （新增，审计遗漏） **Clang 的 `FYX_ISA_BEGIN` 宏参数不展开** | 🔴 **致命：所有 Clang 构建的 SIMD 全部失效** | 见下方 0.2 |

### 0.1 关于条目 3：真因是填充浪费，不是降频

我实测了 AVX2 与 AVX-512 网络排序在小规模下的对比（uint32，取 5 轮最小值，单位 ns）：

| n | AVX2 | AVX-512 | 胜者 |
|---|---|---|---|
| 8 | **7.5** | 32.0 | AVX2（**4.3×**）|
| 16 | **16.7** | 26.8 | AVX2 |
| 24 | **35.5** | 38.5 | AVX2 |
| 32 | **36.3** | 36.9 | 持平 |
| 48 | 77.0 | **68.8** | AVX-512 |
| 64 | 79.2 | **72.7** | AVX-512 |

结论确实是"小规模该用 AVX2"，但**机制必须纠正**：

> 注意 AVX-512 在 **n=8 时（32.0ns）比 n=16 时（26.8ns）还慢**。
> 如果是降频，耗时不可能随数据量增加而下降。

真因：AVX-512 的 `kLanes = 16`，排 8 个元素也要把整条 512-bit 向量用 `sentinel` 填满，然后跑**完整的 16 元素 bitonic 网络**——一半的比较器在给哨兵值做无用功。这是**填充浪费（padding waste）**。

这个区别直接决定修法：
- 若是降频 → 需要按"频率/L3"建模，即审计建议的 `avx512_threshold(n)` 与频率挂钩；
- 若是填充浪费 → 只需**按 lane 数做编译期选择**：`n <= 2*L_avx2` 时选 AVX2。零运行时开销，无需任何频率探测。

**采纳后者。** 频率建模不仅解决不了这个问题，还会引入不可移植的 MSR 读取。
（补充：Ice Lake-SP 的 AVX-512 降频远轻于 Skylake-SP；在轻量 shuffle 型负载上基本不触发 license-based downclocking。降频是真实存在的现象，但**不是这里的瓶颈**。）

### 0.2 新发现的致命 bug：Clang 分支宏参数不展开

`parts/07_simd_net.hpp:76`：

```c
#  define FYX_ISA_BEGIN(isa)  \
      _Pragma("clang attribute push (__attribute__((target(isa))), apply_to = function)")
```

`isa` 出现在**字符串字面量内部**，预处理器不会替换它。实测验证：

```
输入:  A("avx2")     // 现有写法
输出:  #pragma clang attribute push (__attribute__((target(isa))), ...)   ← 字面量 isa
输入:  B("avx2")     // 经 STRINGIFY 间接展开
输出:  #pragma clang attribute push (__attribute__((target("avx2"))), ...) ← 正确
```

后果：**Clang 下所有 AVX2/AVX-512 内核要么编译失败，要么被静默降级**。审计清单声称覆盖 "Clang 5+"，但这条比清单上任何一条都严重。

修法（一行）：

```c
#  define FYX_ISA_BEGIN(isa)                                     \
      FYX_DIAG_PUSH_SIMD                                         \
      _Pragma(FYX_STRINGIFY(                                     \
          clang attribute push(__attribute__((target(isa))),     \
                               apply_to = function)))
```

`FYX_STRINGIFY` 已存在于 `parts/01_config.hpp`，GCC 分支也正是这么用的。

### 0.3 关于条目 9：编译时间属实，I-Cache 论据不成立

实测（g++ 12.2）：

| 目标 | -O2 | -O3 -march=native |
|---|---|---|
| 仅 include 头文件（不实例化） | 1.49 s | 1.25 s |
| `t_net.cpp`（实例化 8 套 Ops） | **6.97 s** | **7.69 s** |

编译时间问题**确实存在**，值得优化。但"I-Cache 压力大"不成立：运行时 CPU 分发只会进入**一条** ISA 路径，另外三条的指令页从不被取指，不占用 I-Cache。真实代价是**编译时间**和**二进制体积**（`t_net.o` text 段 100 KB）。

⚠️ 另需注意：**四份重复是被迫的，不是疏忽**。此前已用 `/tmp/t8.cpp` 证明——**target 属性在模板"定义处"绑定，而非实例化处**。在基线区域定义、在 AVX-512 区域实例化，会触发 `-Werror=psabi`。所以不能简单"合并成一份模板"，只能减少**实例化数量**（见 Module E）。

---

## 第 1 节：总体架构图（调度流程）

```
                    ┌─────────────────────────────────────┐
                    │  入口 API                            │
                    │  sort(c) / sort(p,n) / sort(f,l)     │
                    │  + Options{threads, gpu, stability}  │
                    └───────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  ① 类型萃取（编译期，零开销）                 │
              │  is_arithmetic / has_radix_traits           │
              │  is_trivially_copyable / sizeof(T)          │
              │  comparator == std::less?                   │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  ② 环境探测（进程内一次，缓存于 static）      │
              │  CpuInfo: CPUID → SSE42/AVX2/AVX512/NEON    │
              │  EnvInfo: get_effective_cores()  ← Module A │
              │           cgroup v2 → v1 → affinity → env   │
              │           cache_size(L2/L3)                 │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  ③ 数据分布采样（O(√n) 上限 4096 个样本）     │
              │  sorted / reverse / low-cardinality /       │
              │  sawtooth / random                          │
              │  ※ 已排序 & 逆序 → 直接返回 / reverse，O(n)  │
              └─────────────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────▼────────────────────────────┐
        │  ④ 算法选择器                                            │
        └───────────────────────────┬────────────────────────────┘
                                    │
   ┌────────────┬───────────────────┼───────────────┬─────────────┐
   │            │                   │               │             │
   ▼            ▼                   ▼               ▼             ▼
 n<=64      n<=1024            数值+默认比较器    非数值/自定义   需要稳定
   │            │                   │               │             │
   ▼            ▼                   ▼               ▼             ▼
SIMD网络    分支无关插排/         并行 LSD 基数     块采样排序    向量化归并
   │        小网络组合             （Module F）    （Module C）  （Module D）
   │                                  │               │             │
   │                                  ▼               ▼             │
   │                            n>=并行阈值?     n>=采样阈值?        │
   ▼                                  │               │             │
 ⑤ ISA 选择器（带 lane 门槛，Module B）                              │
   n <= 2*L_avx2 → AVX2；否则 → 最高可用 ISA                        │
   MSVC: 函数指针表；GCC/Clang: target 区域                          │
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  ⑥ 并行策略（自适应配额）                    │
              │  effective_cores = Module A 的结果          │
              │  threshold = max(32768, L2_bytes/sizeof(T)) │
              │  cores==1 → 强制串行（容器限频保护）          │
              │  否则 fork_join → Chase-Lev 工作窃取         │
              └────────────────────────────────────────────┘
```

**关键设计原则**：①②在编译期或进程启动期完成，③的采样成本 O(√n) 且上限 4096，④⑤⑥全部是分支判断。整条链路对 n=10⁶ 的额外开销 < 0.1%。

---

## 第 2 节：关键模块修改

### Module A：容器感知的线程池

**实测发现：审计给出的路径在真实容器里就是错的。**

```
$ cat /sys/fs/cgroup/cpu.max
cat: /sys/fs/cgroup/cpu.max: No such file or directory     ← 审计指定的路径

$ cat /proc/self/cgroup
0::/user                                                    ← 必须先读这里

$ cat /sys/fs/cgroup/user/cpu.max
max 100000                                                  ← 真实位置
```

cgroup v2 下 `cpu.max` 位于**当前 cgroup 的相对路径**下，且 cpu 控制器可能只在子层级被委派（delegate）。直接读根路径在本机 100% 失败。

**并且审计遗漏了 K8s 最常见的限制方式**：`cpuset`（绑核）。K8s 的 Guaranteed QoS Pod 用的是 cpuset 而非 cfs quota，此时 `cpu.max` 是 `max`，但进程只能跑在 2 个核上。**必须同时查 `sched_getaffinity`**。

正确的优先级链（四级回退）：

```
FYX_CPU_LIMIT 环境变量（显式覆盖，最高优先级）
   ↓ 未设置
sched_getaffinity()            ← cpuset / taskset / K8s Guaranteed
   ↓ 取 min
cgroup v2: /proc/self/cgroup → 逐级向上找 cpu.max
cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us ÷ cpu.cfs_period_us
   ↓ 全部失败
std::thread::hardware_concurrency()
```

函数原型与骨架：

```cpp
namespace fyx { namespace detail {

/// cgroup 配额解析结果。quota <= 0 表示"无限制/无法解析"。
struct CpuQuota {
    double cores = -1.0;   ///< 允许的 CPU 核心当量，如 1.5
    bool   valid() const noexcept { return cores > 0.0; }
};

/// 解析 cgroup v2：先读 /proc/self/cgroup 得到相对路径，再逐级向上查找
/// cpu.max（控制器可能只在上层被委派）。格式为 "<quota> <period>" 或
/// "max <period>"。
CpuQuota parse_cgroup_v2_quota() noexcept;

/// 解析 cgroup v1：cpu.cfs_quota_us / cpu.cfs_period_us。
/// quota == -1 表示无限制。
CpuQuota parse_cgroup_v1_quota() noexcept;

/// 亲和性掩码中可用的 CPU 数（cpuset / taskset / K8s Guaranteed QoS）。
/// Linux: sched_getaffinity；Windows: GetProcessAffinityMask；
/// FreeBSD: cpuset_getaffinity。失败返回 0。
unsigned affinity_cpu_count() noexcept;

/// 最终生效核心数。四级回退，结果缓存于函数内 static（只计算一次）。
/// 保证返回值 >= 1。
unsigned get_effective_cores() noexcept;

}} // namespace
```

`get_effective_cores()` 实现骨架：

```cpp
inline unsigned get_effective_cores() noexcept {
    static const unsigned cached = [] () -> unsigned {
        // 1. 显式覆盖
        if (const char* e = std::getenv("FYX_CPU_LIMIT")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v > 0) return clamp_cores(static_cast<unsigned>(v));
        }
        unsigned best = 0;

        // 2. 亲和性（cpuset / taskset）
        if (const unsigned a = affinity_cpu_count())
            best = a;

        // 3. cgroup 配额，向上取整但至少 1
        //    1.5 核 -> 2 线程（配额是时间片，不是并发度上限）
        CpuQuota q = parse_cgroup_v2_quota();
        if (!q.valid()) q = parse_cgroup_v1_quota();
        if (q.valid()) {
            const unsigned qc = static_cast<unsigned>(q.cores + 0.5) ?: 1u;
            best = best ? (qc < best ? qc : best) : qc;
        }

        // 4. 兜底
        if (!best) best = std::thread::hardware_concurrency();
        return clamp_cores(best ? best : 1u);
    }();
    return cached;
}
```

**⚠️ 一处必须谨慎的语义**：CFS 配额是**时间片**，不是**并发度上限**。`cpu.max = 150000 100000`（1.5 核）的容器里开 2 个线程是合理的（各跑 75% 时间片）；开 32 个线程才会因为频繁被 throttle 而崩盘。所以配额向上取整 + 与亲和性取 min，而不是简单截断。

### Module B：跨平台 SIMD 调度器（MSVC 是重点）

MSVC 没有 `__attribute__((target))`，也没有等价的 per-function pragma。业界唯一可靠方案是**分离编译单元 + 运行时函数指针表**。但这与"单头文件、只需 `#include`"的硬约束直接冲突。

**权衡后的双模方案**：

**模式 1（默认，纯头文件）**：MSVC 下依据 `/arch` 编译出的**最高** ISA 提供内核，运行时 CPUID 检查决定是否调用。这保留了"只需 include"的体验，代价是用户不加 `/arch:AVX2` 就吃不到 AVX2。这是 MSVC 的固有限制，**必须在 README 显式说明**，不能假装无事发生。

**模式 2（可选，多 TU）**：定义 `FYX_MSVC_MULTI_TU`，配合提供的 CMake 片段，把 4 份内核编译进 4 个 `.obj`，运行时装配函数表。这条路能拿到完整的运行时分发，代价是不再是"单文件"。

```cpp
// ── 统一的分发层（两种模式共用同一套调用点）─────────────────
enum class IsaLevel : unsigned { Scalar = 0, Sse42 = 1, Avx2 = 2, Avx512 = 3 };

struct SortKernelTable {
    void (*sort_u32)(std::uint32_t*, std::size_t);
    void (*sort_u64)(std::uint64_t*, std::size_t);
    void (*network_u32)(std::uint32_t*, std::size_t);
    // …
    IsaLevel level;
};

/// 进程内构建一次。GCC/Clang 下各槽位指向 target 区域内的函数；
/// MSVC 模式 2 下指向各 TU 导出的符号；MSVC 模式 1 下指向本 TU 的实现。
const SortKernelTable& kernel_table() noexcept;

/// 测试/调优用：强制降级 ISA，绝不允许升到硬件不支持的等级。
bool force_isa_level(IsaLevel lvl) noexcept;
```

**并在选择器里加入 lane 门槛（第 0.1 节的实测结论）**：

```cpp
/// 小数组要按 lane 数选 ISA，不是按频率。
/// 元素少于 AVX2 两个向量时，AVX-512 的一半比较器在给哨兵做无用功。
template <typename Key>
constexpr IsaLevel isa_for_network(std::size_t n, IsaLevel hw) noexcept {
    constexpr std::size_t kAvx2Lanes = 32 / sizeof(Key);   // u32 -> 8
    if (hw == IsaLevel::Avx512 && n <= kAvx2Lanes * 2)     // u32 -> n <= 16
        return IsaLevel::Avx2;
    return hw;
}
```

按实测表，此改动让 n=8 提速 **4.3×**、n=16 提速 **1.6×**，且 n>=48 时仍走 AVX-512。**纯编译期常量比较，零运行时开销。**

### Module C：块采样分类器（非数值类型）

移植 ips4o 的分块思想，比较器替换为分支无关谓词 + 双指针置换。

```cpp
/// 隐式置换的块采样排序（in-place，O(1) 额外空间除去块缓冲）。
/// 步骤：
///   1. 采样 k*log(n) 个元素，排序后等距取 kSampleBuckets-1 个 splitter
///   2. 用隐式二叉搜索树分类（分支无关：用 cmov 累加而非跳转）
///   3. 每桶一个块缓冲，满则写回；写回位置由前缀和确定
///   4. 尾部块做一次置换收尾，递归处理各桶
template <typename It, typename Comp>
void sample_sort(It first, It last, Comp comp, unsigned depth);

/// 分支无关分类：返回元素所属桶号。
/// 隐式树 => 无指针追逐；比较结果直接算入索引 => 无分支预测失败。
template <typename T, typename Comp>
FYX_FORCE_INLINE unsigned classify(const T& x, const T* tree,
                                   unsigned log_buckets, Comp comp) {
    unsigned b = 1;
    for (unsigned l = 0; l < log_buckets; ++l)
        b = 2 * b + static_cast<unsigned>(comp(tree[b], x));  // 无分支
    return b - (1u << log_buckets);
}
```

**关键**：`classify` 的循环次数是编译期常量（`log_buckets = 8`），可完全展开；`comp(tree[b], x)` 的结果直接参与地址计算，编译器会生成 `setcc`/`cmov` 而非 `jcc`。这正是 ips4o 相对 `std::sort` 的主要优势来源。

### Module D：向量化归并（稳定排序）

先纠正审计的一处事实错误：**ips4o 不是稳定排序**，不存在"ips4o 的旋转归并"可供对标。稳定排序的正确对标物是 `std::stable_sort`（GCC 的 `__inplace_stable_sort` / 带缓冲归并）和 `__gnu_parallel::stable_sort`。

审计提到用 `vpminsb`/`vpmaxsb` 做块级挑选——`vpminsb` 是 **8 位**有符号 min，用于归并 32/64 位键是错的指令。正确做法是**双调归并网络（bitonic merge）**：

```cpp
/// 两个已排序的 L 元素向量 -> 合并为排序的 2L 序列（占用两个向量）。
/// 原理：反转 b，与 a 做 min/max 得到双调序列，再做 log(L) 轮清理。
template <typename Ops>
FYX_FORCE_INLINE void merge_2v(typename Ops::Vec& a, typename Ops::Vec& b) {
    b = Ops::reverse(b);
    typename Ops::Vec lo = Ops::min(a, b);
    typename Ops::Vec hi = Ops::max(a, b);
    a = lo; b = hi;
    bitonic_cleanup<Ops>(a);   // log(L) 轮
    bitonic_cleanup<Ops>(b);
}
```

**稳定性怎么保证**：SIMD min/max 会丢失相等元素的原始顺序。方案是**索引打包**——对 `sizeof(T) <= 4` 且 n < 2³² 的情形，把 `(key, index)` 打包成 64 位整数（key 在高位，index 在低位）后排序，相等 key 自动按 index 升序，天然稳定；排完再按 index 回填。对不可打包的类型，退回带缓冲归并，但用 SIMD 加速**块内**排序（run generation）。

### Module E：模板实例化瘦身（对应审计条目 9）

不能合并 target 区域（会触发 `-Werror=psabi`，已证明），但可以砍掉不必要的实例化：

1. **按 lane 门槛裁剪**（Module B 的副产品）：既然 `n <= 16` 的 u32 永远走 AVX2，AVX-512 的 `V=1` 实例可以不生成。
2. **`extern template` 声明** + 可选的 `FYX_SEPARATE_COMPILATION` 模式，把重型实例化收进一个 TU。
3. 用 `if constexpr` 剪掉 `L*V > 64` 的死分支（部分已做）。

预期把 `t_net` 的 7 s 编译时间压到 4 s 量级。**这是编译期优化，不影响运行时性能。**

### Module F：类型覆盖扩展（对应审计条目 8）

| 类型 | 现状 | 方案 |
|---|---|---|
| `int16/uint16` | 掉入比较排序 | 加 `Ops16`（AVX-512 每向量 32 lane，网络更划算）；基数排序 2 pass |
| `int8/uint8` | 掉入比较排序 | **计数排序**，256 桶单遍，比任何比较排序快一个量级 |
| `__int128` | 掉入比较排序 | 基数排序 16 pass；网络用两个 64 位 lane 组合比较 |
| `float16/bfloat16` | 掉入比较排序 | 编译期检测 `__fp16`/`_Float16`；按位序编码后走 u16 基数 |
| `std::array<uint8,16>`（哈希） | 掉入比较排序 | MSD 基数，逐字节；这是审计"魔鬼基准"里的 128 位哈希项 |

`bool` 继续排除在 `RadixTraits` 之外（已是刻意决定）。

### Module G：ScratchLease 嵌套复用（对应审计条目 10）

现状：`if (!a.in_use())` 失败即 `aligned_malloc`。递归排序每层都可能分配。

改为 **bump 指针栈式 arena**：

```cpp
class ScratchArena {
    // 单调递增的 bump 指针；lease 析构时回退到保存的水位线。
    // 嵌套 lease 各自占用一段，互不冲突，全程零 malloc。
    char*       base_ = nullptr;
    std::size_t cap_  = 0;
    std::size_t top_  = 0;      ///< 当前水位线
public:
    void* push(std::size_t bytes, std::size_t align) noexcept;
    void  pop(std::size_t saved_top) noexcept { top_ = saved_top; }
};
```

`ScratchLease` 记录 `saved_top_`，析构时 `pop`。只有 arena 容量不足时才 grow（几何增长），**递归路径上一次 malloc 都不会发生**。

### Module H：ARM fence（对应审计条目 7，谨慎处理）

⚠️ **本沙箱是 x86_64，无 aarch64 交叉编译器、无 qemu，此项无法验证。以下是方案，不是结论。**

首先澄清：`pop()` 和 `steal()` 里的 `seq_cst` fence **不能删除**——它正是防止 owner 和 thief 同时取走最后一个元素的机制。删掉会得到一个"在 x86 上看起来能跑、在 ARM 上偶发丢任务"的队列，这比慢更糟。

可安全落地的 ARM 优化：

1. **快路径提前退出**：`steal()` 在 fence **之前**先做一次 relaxed 的空判断，空队列直接返回，完全不执行 fence。窃取失败是常态（本机实测 2M 轮里 99.98% 的 pop 无竞争），这能消掉绝大多数 fence。
2. **`pop()` 单元素快路径**：`bottom - top > 1` 时无需与 thief 竞争，可用 acquire/release 而非 seq_cst。
3. 保留完整 fence 于真正的竞争路径。

```cpp
StealStatus steal(Task& out) {
    // 快路径：空队列不付 fence 代价（ARM 上 dmb ish 尤其贵）
    if (bottom_.load(std::memory_order_relaxed) <=
        top_.load(std::memory_order_relaxed))
        return StealStatus::Empty;
    // …原有 seq_cst 路径…
}
```

**验收要求**：此项必须在真实 ARM 硬件（Apple M 系列 / Graviton）上用 `t_deque_race` 跑通，且需在 relaxed 快路径上补充 ThreadSanitizer 验证。**未经 ARM 实测不得合入。**

---

## 第 3 节：核心伪代码

### 3.1 `sort_dispatch`

```cpp
template <typename It, typename Comp>
void sort_dispatch(It first, It last, Comp comp, const Options& opt) {
    using T = typename std::iterator_traits<It>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);

    // ── 平凡情形 ──────────────────────────────────────────
    if (n < 2) return;

    // ── ① 编译期类型决策 ──────────────────────────────────
    constexpr bool kContig    = is_contiguous_iterator<It>::value;
    constexpr bool kDefaultCmp= is_default_less<Comp, T>::value;
    constexpr bool kRadixable = has_radix_traits<T>::value;

    // ── ② 环境（进程内缓存）───────────────────────────────
    const CpuInfo&  cpu   = cpu_info();
    const unsigned  cores = opt.threads ? opt.threads : get_effective_cores();

    // ── ③ 小数组：SIMD 网络，禁止插入排序 ──────────────────
    if (n <= kNetworkMax && kContig && kDefaultCmp && kRadixable) {
        // lane 门槛：小 n 用 AVX2 而非 AVX-512（实测 n=8 快 4.3×）
        const IsaLevel lvl = isa_for_network<radix_key_t<T>>(n, cpu.level);
        network_sort_dispatch<T>(&*first, n, lvl);
        return;
    }

    // ── ④ 数据分布探测（O(min(sqrt(n), 4096))）─────────────
    const Distribution d = probe_distribution(first, last, comp);
    if (d == Distribution::Sorted)   return;                    // O(n) 提前退出
    if (d == Distribution::Reverse){ std::reverse(first,last); return; }
    // 极低基数 -> 三路划分一遍即可解决，避免基数排序的多趟浪费
    if (d == Distribution::VeryLowCardinality && n > kRadixThreshold) {
        three_way_partition_sort(first, last, comp, cores);
        return;
    }

    // ── ⑤ 稳定性需求 ─────────────────────────────────────
    if (opt.stable) { stable_sort_dispatch(first, last, comp, cores); return; }

    // ── ⑥ 主路径选择 ─────────────────────────────────────
    const bool parallel = (cores > 1) && (n >= parallel_threshold<T>());

    if (kContig && kDefaultCmp && kRadixable && n >= kRadixThreshold) {
        if (parallel) parallel_radix_sort(&*first, n, cores);
        else          radix_sort(&*first, n);          // 已实测 int32 6.24x
        return;
    }
    if (parallel && n >= kSampleThreshold) {
        sample_sort(first, last, comp, cores);          // Module C
        return;
    }
    pdqsort(first, last, comp);                         // 已实测 int 2.05x
}
```

**关键点**：所有类型判断是 `constexpr`，不产生运行时分支；分布探测只在 n > kNetworkMax 时进行；"已排序"提前退出让最好情形降到 O(n)。

### 3.2 `avx512_threshold` —— 按实测改为 lane 模型

审计要求"与 L3 缓存和 CPU 频率挂钩"。第 0.1 节的实测证明频率不是这里的变量，**故意不按原样实现**，改为两段式：

```cpp
/// 小规模：lane 门槛（编译期常量，见 Module B 的 isa_for_network）
///
/// 大规模：AVX-512 的收益来自带宽，此时应关注工作集与 L2 的关系，
/// 而非频率。基数排序的实测结论是——瓶颈在 scatter 的输出工作集
/// 超过 L2（此前已用 huge page 实验排除 TLB 假设）。
template <typename Key>
inline std::size_t avx512_min_elements() noexcept {
    // 低于两个 AVX2 向量的元素数，AVX-512 一半 lane 在处理哨兵
    constexpr std::size_t kAvx2Lanes = 32 / sizeof(Key);
    return kAvx2Lanes * 2 + 1;
}

/// 并行阈值：与 L2 挂钩而非写死 32768（对应审计条目 6）
template <typename T>
inline std::size_t parallel_threshold() noexcept {
    const std::size_t l2 = cpu_info().l2_bytes ? cpu_info().l2_bytes
                                               : (1u << 20);
    // 单线程能装进 L2 的规模不值得并行；再乘以核数留出切分空间
    const std::size_t by_cache = l2 / sizeof(T);
    return by_cache > kParallelThreshold ? by_cache : kParallelThreshold;
}
```

### 3.3 `parse_cgroup_cpu_quota` —— Linux 实现壳

```cpp
/// cgroup v2。必须先解析 /proc/self/cgroup 得到相对路径，再逐级向上查找，
/// 因为 cpu 控制器可能只在某一层被委派。
/// 实测：本机 /sys/fs/cgroup/cpu.max 不存在，真实位置是
///       /sys/fs/cgroup/user/cpu.max ("max 100000")。
inline CpuQuota parse_cgroup_v2_quota() noexcept {
    CpuQuota out;
#if FYX_OS_LINUX
    // 1. 取相对路径： "0::/user" -> "/user"
    char rel[PATH_MAX] = {0};
    if (!read_cgroup_relative_path(rel, sizeof rel)) return out;

    // 2. 从最深层向上逐级尝试 <mount>/<rel>/cpu.max
    char path[PATH_MAX];
    for (;;) {
        if (snprintf(path, sizeof path, "/sys/fs/cgroup%s/cpu.max", rel) > 0) {
            char buf[64];
            if (read_small_file(path, buf, sizeof buf)) {
                // 格式: "<quota> <period>" 或 "max <period>"
                if (std::strncmp(buf, "max", 3) == 0) return out;  // 无限制
                long q = 0, p = 0;
                if (std::sscanf(buf, "%ld %ld", &q, &p) == 2 && q > 0 && p > 0) {
                    out.cores = static_cast<double>(q) / static_cast<double>(p);
                    return out;
                }
            }
        }
        if (!strip_last_path_component(rel)) break;   // 到根仍未找到
    }
#endif
    return out;
}

/// cgroup v1。路径固定，但 quota == -1 表示无限制。
inline CpuQuota parse_cgroup_v1_quota() noexcept {
    CpuQuota out;
#if FYX_OS_LINUX
    long q = read_long_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",  -2);
    long p = read_long_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", -2);
    if (q > 0 && p > 0) out.cores = double(q) / double(p);
#endif
    return out;
}
```

**实现注意**：全部使用 `open/read/close` 而非 `<fstream>`——这些函数会在静态初始化期被调用，iostream 的初始化顺序不可依赖；且必须 `noexcept`。

---

## 第 4 节：构建系统（CMake）适配

### 4.1 MSVC 多 TU 函数表（模式 2）

```cmake
# ── 默认：纯头文件接口库（单文件约束下的主路径）────────────────
add_library(fyx_sort INTERFACE)
target_include_directories(fyx_sort INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(fyx_sort INTERFACE cxx_std_17)

# ── 可选：MSVC 多 TU 运行时分发 ───────────────────────────────
option(FYX_MSVC_MULTI_TU "MSVC: per-ISA object files + runtime table" OFF)

if(FYX_MSVC_MULTI_TU AND MSVC)
    # 每个 ISA 一个 TU，各自用不同 /arch 编译
    foreach(isa SCALAR SSE42 AVX2 AVX512)
        add_library(fyx_kernel_${isa} OBJECT src/kernel_tu.cpp)
        target_compile_definitions(fyx_kernel_${isa}
            PRIVATE FYX_TU_ISA=FYX_ISA_${isa})
        target_include_directories(fyx_kernel_${isa} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()

    # /arch 是全 TU 生效的，所以必须分 TU
    target_compile_options(fyx_kernel_AVX2   PRIVATE /arch:AVX2)
    target_compile_options(fyx_kernel_AVX512 PRIVATE /arch:AVX512)
    # SSE42/SCALAR 用默认 /arch:SSE2，靠内部 intrinsics 保证不越级

    add_library(fyx_sort_msvc STATIC
        src/dispatch.cpp
        $<TARGET_OBJECTS:fyx_kernel_SCALAR>
        $<TARGET_OBJECTS:fyx_kernel_SSE42>
        $<TARGET_OBJECTS:fyx_kernel_AVX2>
        $<TARGET_OBJECTS:fyx_kernel_AVX512>)
    target_compile_definitions(fyx_sort_msvc PUBLIC FYX_MSVC_MULTI_TU=1)
endif()
```

⚠️ **MSVC 的一个硬坑**：`/arch:AVX512` 允许编译器在**整个 TU**自由发射 AVX-512 指令，包括编译器自动生成的 memcpy/初始化代码。所以 `dispatch.cpp` **必须**用基线 `/arch` 编译，绝不能与内核 TU 合并——否则分发代码本身会在老 CPU 上触发 `#UD`（非法指令）崩溃。这是 MSVC 运行时分发最常见的翻车点。

### 4.2 `configure_file` 生成 config.h

```cmake
include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "${FYX_AVX512_FLAG}")
check_cxx_source_compiles("
    #include <immintrin.h>
    int main(){
        __m512i a = _mm512_set1_epi32(1);
        __m512i c = _mm512_conflict_epi32(a);   // 需要 AVX512CD
        return _mm512_reduce_add_epi32(c);
    }" FYX_COMPILER_HAS_AVX512)

check_cxx_source_compiles("
    #include <arm_neon.h>
    int main(){ uint32x4_t v = vdupq_n_u32(1); return vgetq_lane_u32(v,0); }
" FYX_COMPILER_HAS_NEON)

configure_file(cmake/fyx_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/fyx_config.h @ONLY)
```

```c
/* cmake/fyx_config.h.in */
#pragma once
#cmakedefine01 FYX_COMPILER_HAS_AVX512
#cmakedefine01 FYX_COMPILER_HAS_NEON

#if !FYX_COMPILER_HAS_AVX512 && defined(FYX_REQUIRE_AVX512)
#  error "FYX_REQUIRE_AVX512 was requested but this compiler cannot build \
AVX-512 kernels. Upgrade to MSVC 2019 16.3+/GCC 7+/Clang 5+, or drop the flag."
#endif
```

**注意**：探测代码里特意用了 `_mm512_conflict_epi32`（AVX512CD）而非最基础的 AVX512F 指令——因为基数排序的 SIMD 直方图依赖 CD 子集，只测 F 会漏判。

---

## 第 5 节：三个最小可测 Demo

### Demo 1：容器内核心数感知

```cpp
// demo_cores.cpp —— 在容器内运行，验证不会超卖
#include "fyx_sort.hpp"
#include <cstdio>
int main() {
    printf("hardware_concurrency : %u\n", std::thread::hardware_concurrency());
    printf("affinity_cpu_count   : %u\n", fyx::detail::affinity_cpu_count());
    auto v2 = fyx::detail::parse_cgroup_v2_quota();
    auto v1 = fyx::detail::parse_cgroup_v1_quota();
    printf("cgroup v2 quota      : %s\n", v2.valid() ? std::to_string(v2.cores).c_str() : "n/a");
    printf("cgroup v1 quota      : %s\n", v1.valid() ? std::to_string(v1.cores).c_str() : "n/a");
    printf("EFFECTIVE CORES      : %u\n", fyx::detail::get_effective_cores());
    printf("pool workers         : %u\n", fyx::detail::global_pool().nworkers());
}
```

验收脚本：

```bash
# 限制为 1.5 核，期望 effective cores <= 2 而非宿主机核数
docker run --cpus=1.5 fyx-demo ./demo_cores
# 绑到 2 个核（K8s Guaranteed 等价），期望 == 2
taskset -c 0,1 ./demo_cores
# 显式覆盖
FYX_CPU_LIMIT=3 ./demo_cores
```

### Demo 2：运行时 ISA 切换

```cpp
// demo_isa.cpp
#include "fyx_sort.hpp"
int main() {
    std::vector<std::uint32_t> v(1 << 20);
    std::mt19937 rng(42);
    for (auto& x : v) x = rng();

    for (auto lvl : {IsaLevel::Scalar, IsaLevel::Sse42,
                     IsaLevel::Avx2,   IsaLevel::Avx512}) {
        if (!fyx::detail::force_isa_level(lvl)) {
            printf("%-8s : unsupported on this CPU, skipped\n", isa_name(lvl));
            continue;                       // 绝不在不支持的硬件上执行
        }
        auto c = v;
        auto t0 = Clk::now();
        fyx::sort(c);
        printf("%-8s : %8.2f ms  sorted=%d\n", isa_name(lvl),
               ms(Clk::now() - t0), (int)std::is_sorted(c.begin(), c.end()));
    }
}
```

### Demo 3：魔鬼基准套件

```cpp
// bench_devil.cpp —— 覆盖审计要求的全部 6 类数据
enum class Dataset {
    StringShort, StringLongSharedSuffix,   // 字符串
    Struct3Byte,                            // 非 2 幂结构体
    Cardinality2,                           // 极低基数
    Reverse, Sorted, Sawtooth,              // 病态序
    FloatNaNZero,                           // NaN / ±0
    Hash128,                                // std::array<uint8_t,16>
};
// 每组: n ∈ {1e3, 1e4, 1e5, 1e6, 1e7}
// 输出: 各实现耗时 + 几何平均 + 与 std::sort 的比值
// 硬性断言: 任何 (数据集, n) 组合下都不得慢于 std::sort 的 1.0x
//           —— 这条断言就是"不许某种数据让库退化"的可执行版本
```

配套脚本 `scripts/run_devil.sh`：跑满矩阵、输出 CSV、计算几何平均、生成对比曲线数据。

**关于与 ips4o 对比曲线**：本沙箱**无网络**，无法拉取 ips4o/TBB/vqsort 源码，因此**无法产出真实对比数据**。脚本会预留 `--with-ips4o=<path>` 参数，在有网络的环境中补测。**我不会编造对比数字。**

---

## 第 6 节：预期性能增益表

⚠️ **区分"实测"与"预期"**。下表中标注实测的行有本轮数据支撑；标注预期的行是基于机制的估计，**必须在合入前用 Demo 3 验证**。

| 模块 | 场景 | 增益 | 依据 |
|---|---|---|---|
| **B（lane 门槛）** | u32, n=8 | **4.3×** | ✅ **实测** 32.0ns → 7.5ns |
| **B（lane 门槛）** | u32, n=16 | **1.6×** | ✅ **实测** 26.8ns → 16.7ns |
| **B（lane 门槛）** | u32, n≥48 | 无变化（仍走 AVX-512） | ✅ 实测 |
| **0.2（Clang 修复）** | 全部 Clang 构建 | **从"SIMD 失效"到可用** | ✅ 预处理实测 |
| **1（MSVC 分发）** | Windows，未加 /arch | **标量 → AVX2/512** | 机制确定，量级取决于类型 |
| **C（块采样）** | 字符串 / 大结构体，n≥1e5 | 预期 1.5–2.5× vs pdqsort | ips4o 公开数据的量级；**待验证** |
| **F（int8 计数排序）** | uint8, n≥1e5 | 预期 5–15× | 单遍 O(n) vs O(n log n) |
| **F（int16 基数）** | uint16, n≥1e5 | 预期 3–6× | 2 pass 基数 |
| **A（cgroup 感知）** | 1.5 核容器内 | 预期消除 throttle 抖动 | **不是提速，是防崩**；超卖场景下尾延迟改善最明显 |
| **G（栈式 arena）** | 深递归排序 | 预期递归路径 malloc 归零 | 分配次数可直接计数验证 |
| **E（实例化瘦身）** | 编译期 | 预期 7s → 4s | ✅ 基线实测 6.97s |
| **H（ARM fence）** | Apple M / Graviton | **未知** | ⚠️ 无 ARM 硬件，不给数字 |
| **D（向量化归并）** | stable_sort | 预期 1.5–3× vs std::stable_sort | 对标物是 std::stable_sort，**不是 ips4o** |

已有实测基线（本轮之前）供参照：

| 已实现部分 | 相对 std::sort |
|---|---|
| 基数排序 int32 10M | **6.24×** |
| 基数排序 float 10M | **6.92×** |
| 基数排序 int64 10M | 2.52× |
| pdqsort int 5M | 2.05× |

---

## 第 7 节：实施优先级

按"修复致命 → 补齐能力 → 优化"排序：

| 优先级 | 项目 | 理由 | 可在本沙箱验证 |
|---|---|---|---|
| **P0** | 0.2 Clang 宏修复 | 一行，当前所有 Clang 构建的 SIMD 都是坏的 | ✅ |
| **P0** | Module B MSVC 分发 | Windows 平台可用性 | ❌ 无 MSVC |
| **P1** | Module B lane 门槛 | 一行 constexpr，实测 4.3× | ✅ |
| **P1** | Module A 容器感知 | 云原生环境正确性 | ✅ 本机 cgroup v2 |
| **P1** | Module C 块采样 | 非数值类型的核心竞争力 | ✅ |
| **P2** | Module F 类型扩展 | 覆盖面 | ✅ |
| **P2** | Module G 栈式 arena | 递归场景 | ✅ |
| **P2** | Module D 稳定排序 | 功能补齐 | ✅ |
| **P3** | Module E 编译瘦身 | 开发体验 | ✅ |
| **P3** | Module H ARM fence | **必须在真实 ARM 上验证后才可合入** | ❌ |

---

## 第 8 节：关于"宇宙最强"目标的诚实评估

必须说清楚三件事，否则这份方案会给出错误预期：

**1. 数值类型上"吊打"是有据的，泛型类型上还没有。**
基数排序 int32 6.24× 是实测。但审计的核心诉求——**在字符串/结构体上碾压 ips4o**——目前连采样排序都还没写完（Module C）。ips4o 是经过多年打磨的高质量实现，"预期 1.5–2.5×"是基于机制的乐观估计，**在跑出真实对比数据前，不应对外宣称超越**。

**2. 本沙箱无法完成验收。**
- 无网络 → 拿不到 ips4o / TBB / vqsort 源码，**无法产出任何真实对比曲线**；
- 2 vCPU → 多线程 ≥30 GB/s 的目标**无法验证**；
- x86_64 且无交叉编译器 → ARM/NEON、MSVC、MinGW 路径**全部无法编译验证**；
- 无 GPU → GPU 路径只能是骨架。

**3. "在所有数据上都最快"在理论上不可达。**
更务实、也更有价值的目标是 Demo 3 里那条硬断言：**任何数据分布下都不劣于 `std::sort`，且在主流场景显著更快**。一个"从不退化"的库，比一个"平均很快但某类数据上崩盘"的库更有资格成为通用标准。这也正是审计条目里"不许出现某种数据让库退化为 O(n²)"的正确表述方式。

---

## 附：本轮实测数据汇总（可复现）

```
环境: Intel Xeon Ice Lake-SP (family 6 model 106), 2 vCPU, g++ 12.2.0, cgroup v2

[LICENSE]        Apache License 2.0  (审计条目 2 不成立)
[cgroup]         /sys/fs/cgroup/cpu.max        -> 不存在
                 /proc/self/cgroup             -> "0::/user"
                 /sys/fs/cgroup/user/cpu.max   -> "max 100000"   ← 真实位置
                 sched_getaffinity             -> 2
[AVX2 vs AVX512 网络, uint32, ns]
                 n=8   AVX2 7.5   AVX512 32.0   (AVX2 快 4.3x)
                 n=16  AVX2 16.7  AVX512 26.8
                 n=24  AVX2 35.5  AVX512 38.5
                 n=32  AVX2 36.3  AVX512 36.9
                 n=48  AVX2 77.0  AVX512 68.8   (AVX512 反超)
                 n=64  AVX2 79.2  AVX512 72.7
[编译时间]        头文件 only     -O2 1.49s   -O3 -march=native 1.25s
                 t_net.cpp       -O2 6.97s   -O3 -march=native 7.69s
                 t_net.o text 段 100526 bytes
[Clang 宏展开]    现有写法 -> target(isa)      ← 字面量，未展开，BUG
                 STRINGIFY -> target("avx2")   ← 正确
```
