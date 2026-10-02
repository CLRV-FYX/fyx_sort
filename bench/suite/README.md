# fyx 基准测评包

一键在你的机器上完成：fyx 正确性测试 → 构建全部对手 → 交错多轮测评 → 生成报告 → 打包结果。

## 用法

| 系统 | 首次/更新 | 开始测评 |
|---|---|---|
| Linux / macOS / WSL / MSYS2 | `./update.sh`（可选，拉取最新 fyx 与测评框架） | `./start.sh` |
| Windows（cmd / 双击） | `update.cmd` | `start.cmd` |

结束后会打印 `fyx_results_*.tar.gz` 的路径，**把这个文件发回即可**。

常用参数（直接加在 start 后面）：

- `--profile=quick`：1e4 / 1e6，3 轮，快速冒烟。
- 默认 `--profile=full`：1e3 / 1e4 / 1e5 / 1e6 / 1e7，5 轮（取中位数）。
- `--profile=huge`：再加 1e8（需 ≥16 GB 可用内存）。
- `--algos=fyx,fyx_par,ips4o_par`、`--skip=rust_stable`、`--types=i32,str`、`--dists=random,zipf`、`--suites=stable`。
- `--resume`：中断后继续上次未完成的测评（已完成的结果文件会跳过）。
- 环境变量：`CXX=clang++`（Linux/macOS 主编译器）、`FYX_CXX=cl`（Windows 主编译器）、`FYX_ALL_COMPILERS=0`（只测主编译器；默认对其它检测到的编译器再做一次 quick 测评）。

## 参测实现

| 名称 | 来源 | 并行 | 稳定 | 说明 |
|---|---|---|---|---|
| fyx / fyx_par / fyx_stable | 本项目 | 串/并 | 稳定版 | 各构建 native 与 portable（无 ISA 标志，fyx 最不利配置） |
| std_sort / std_stable / std_par / std_par_stable | 标准库 | 串/并 | 部分 | `std::execution::par` 若标准库需要 TBB 而未安装则标为不可用 |
| pdqsort | orlp/pdqsort | 否 | 否 | |
| ips4o / ips4o_par | ips4o/ips4o | 串/并 | 否 | 并行版使用 std::thread 后端 |
| ips2ra / ips2ra_par | ips4o/ips2ra | 串/并 | 否 | 基数排序；字符串不适用 |
| vqsort | google/highway | 否 | 否 | 运行时选择最佳 SIMD 目标；kv16 使用其 K64V64 |
| xss | intel/x86-simd-sort | 否 | 否 | 仅 x86，按本机 AVX2/AVX-512 编译 |
| ska_sort | skarupke/ska_sort | 否 | 否 | 基数排序 |
| boost_spreadsort / boost_pdqsort / boost_block_indirect | Boost.Sort 1.86 | 串/并 | 否 | |
| boost_sample_sort / boost_parallel_stable / boost_spinsort / boost_flat_stable | Boost.Sort 1.86 | 串/并 | 是 | 稳定排序对手 |
| crumsort / fluxsort / quadsort | scandum | 否 | flux/quad 稳定 | |
| rust_unstable / rust_stable | Rust 标准库（ipnsort / driftsort） | 否 | 后者稳定 | 需要 rustc；未安装则标为不可用 |

所有对手均以 `-O3` 与本机 ISA（`-march=native`/`-mcpu=native`，MSVC 为 `/arch:AVX2|AVX512`）构建，并启用其自带并行后端；构建失败的对手会在报告中注明原因，不会被静默跳过。

## 测试矩阵

- 不稳定排序套件：i32、u32、i64、u64、f32、f64、str（4–32 字符）、kv16（16 字节记录按 key 比较）。
- 稳定排序套件：rec8（8 字节记录按 key）、kv16、i32、f64、str；结果必须与 `std::stable_sort` 逐元素一致。
- 16 种分布：random、sorted、reverse、nearly_sorted、few_unique16、few_unique256、all_equal、organ_pipe、sorted_runs、rotated、concat2、block_swap、far_swaps、sorted_tail、zipf、sqrt_unique。
- 数据由可移植的 splitmix64 按 (seed, 类型, n, 分布) 生成，所有可执行文件得到完全相同的输入。

## 报告口径（report.md）

- **A 串行对等**：fyx 串行（native）对最佳串行对手。
- **B 全能力**：fyx 最佳（native，串/并行取优）对任意最佳对手。
- **C 最不利**：fyx 最佳（portable，无 ISA 标志）对任意最佳对手（native）。

逐格列出所有 fyx 未取胜的测试格（最差在前）以及所有正确性失败。

## 目录

- `fyx/`：fyx 头文件与测试（由 update 更新）
- `suite/`：测评框架源码（runner、数据生成、每个对手一个翻译单元）
- `third_party/`：固定版本的对手源码（见 `suite/deps.lock`）
- `results/`：每次测评一个目录（report.md、cases.csv、summary.csv、raw/、logs/、env.txt）
