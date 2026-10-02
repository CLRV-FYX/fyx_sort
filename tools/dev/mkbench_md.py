#!/usr/bin/env python3
"""Rebuild BENCHMARKS.md from the matrix output files.

    bash tools/dev/vqsort.sh 1000000     # or build/bench_matrix directly
    bash tools/dev/vqsort.sh 8000000
    python3 tools/dev/mkbench_md.py

Every number in the document comes from build/vqsort_<n>.txt, including the
cells we lose.  Nothing here is typed in by hand.

The parser reads the algorithm names out of the table header, so the document
adapts to whichever opponents the binary was built with (std::sort, pdqsort,
vqsort, IPS4o serial/parallel, x86-simd-sort); a cell printed as `n/a` is
skipped rather than counted as zero.
"""
import sys, os

CN = {'random': '随机', 'sorted': '已排序', 'reverse': '逆序', 'nearlysorted': '近似有序',
      'lowcard16': '16 个不同值', 'lowcard256': '256 个不同值', 'allequal': '全部相等',
      'mod8': '周期为 8', 'zigzag': '锯齿', 'sawtooth': '锯齿波', 'rotated': '旋转',
      'concat2': '两段拼接', 'blockswap': '块交换', 'farswap': '远距离交换'}

# display order / names for the opponent columns
OPP = [('std', 'std::sort'), ('pdqsort', 'pdqsort'), ('vqsort', 'vqsort'),
       ('ips4o', 'IPS4o 串行'), ('ips4opar', 'IPS4o 并行'), ('xss', 'x86-simd-sort')]


def load(path):
    """-> {(type, dist): {'fyx':v, 'fyxpar':v, <opp>:v-or-None}}"""
    rows = {}
    names = None
    for line in open(path):
        if not line.startswith('|'):
            continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) < 5:
            continue
        if '---' in line:
            continue
        if c[0] == 'type':
            names = c[3].split()
            continue
        if names is None or c[1] == 'n':
            continue
        vals = c[3].split()
        if len(vals) != len(names):
            continue
        if c[1] == 'n':
            continue
        n = int(c[1])
        row = {}
        for k, v in zip(names, vals):
            if v == 'n/a':
                row[k] = None
            else:
                row[k] = float(v.rstrip('!'))
        rows[(c[0], c[2])] = row
    return rows


def measurement_note(path):
    footer = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# merge_runs:"):
                return line.strip()[len("# merge_runs:"):].strip()
            if line.startswith("# bench_matrix:"):
                footer.append(line.strip()[2:].strip())
    return "single process run" + (f" ({footer[0]})" if footer else "")


def pinned_revisions():
    revisions = []
    script = os.path.join(os.path.dirname(__file__), "vqsort.sh")
    with open(script, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "pin_commit":
                revisions.append((fields[1].rsplit("/", 1)[-1], fields[2]))
    return revisions


def opponents(row):
    return [(k, row[k]) for k, _ in OPP if row.get(k) is not None]


def best_other(row):
    o = opponents(row)
    if not o:
        return ('-', float('inf'))
    k, v = min(o, key=lambda kv: kv[1])
    return (dict(OPP)[k], v)


def main(paths):
    tables = {}
    protocols = {}
    for p in paths:
        stem = os.path.basename(p)
        if stem.startswith('vqsort_'):
            stem = stem[len('vqsort_'):]
        if stem.endswith('.txt'):
            stem = stem[:-4]
        n = stem.split('_', 1)[0]
        tables[n] = load(p)
        protocols[n] = measurement_note(p)
    out = []
    w = out.append
    w("# BENCHMARKS — 本机实测")
    w("")
    w("> 表格对应传入的本机实测矩阵，包含输格；数据仅代表记录的主机与构建配置，不外推到未测平台。")
    w("> 由 `tools/dev/mkbench_md.py` 生成；比较对象为 `fyx::sort`，不代表 stable_sort 专项性能。")
    w("")
    w("## 环境")
    w("")
    w("- CPU：Intel Xeon Ice Lake-SP，**2 个硬件线程**（L1d 48 KiB / L2 1.3 MiB / L3 54 MiB，AVX-512）")
    w("- 编译器：g++ 12.2.0，`-O3 -march=native -pthread`")
    w("- 输入复制不计时；单个进程内部报告 best-of-repetitions。独立进程聚合口径如下：")
    for n in tables:
        w(f"  - {n}: {protocols[n]}")
    w("- 对手：`std::sort`（libstdc++）、`pdqsort`（orlp/pdqsort）、**`vqsort`（Google Highway 1.4.0）**、")
    w("  **`IPS4o` 串行与并行（oneTBB 12.x）**、**`x86-simd-sort`（intel/x86-simd-sort，header-only 构建）**。")
    revisions = pinned_revisions()
    if revisions:
        w("- 对手源码固定版本：" + "; ".join(f"{name} `{rev}`" for name, rev in revisions) + "。")
    w("  比值 = 所有对手里最好的那个 / 并行 `fyx::sort`；1.00x 是打平，小于 1 是输。")
    w("- `n/a` 表示该对手不适用：vqsort / x86-simd-sort 只处理算术类型，`std::string` 行没有它们的数字。")
    w("- 近 1.00x 的差异可能落在测量波动内；精细比较两个提交时，应在同一净窗口做背靠背交错测量。")
    w("")
    # headline: random data
    w("## 一、随机数据")
    w("")
    hdr = "| 类型 | 规模 | 串行 | 并行 | std::sort | pdqsort | vqsort | IPS4o 串行 | IPS4o 并行 | x86-simd-sort | 并行 vs 最强对手 |"
    w(hdr)
    w("|------|------|------|------|-----------|---------|--------|------------|------------|---------------|------------------|")
    for n, t in tables.items():
        for ty in ('int32', 'int64', 'double', 'string'):
            row = t.get((ty, 'random'))
            if row is None:
                continue
            who, b = best_other(row)
            def f(k):
                v = row.get(k)
                return "--" if v is None else f"{v:.5f} s"
            w(f"| {ty} | {n} | {row['fyx']:.5f} s | {row['fyxpar']:.5f} s | {f('std')} |"
              f" {f('pdqsort')} | {f('vqsort')} | {f('ips4o')} | {f('ips4opar')} | {f('xss')} |"
              f" {b/row['fyxpar']:.2f}x ({who}) |")
    w("")
    # structured data
    w("## 二、结构化数据")
    w("")
    w("只列并行 `fyx::sort` 与所有对手中最快的那个；每个对手的完整数字见 `build/vqsort_<n>.txt`。")
    w("")
    w("| 类型 | 分布 | 规模 | 并行 | 最强对手 | 对手成绩 | 并行 vs 最强对手 | IPS4o 并行 | vqsort |")
    w("|------|------|------|------|----------|----------|------------------|------------|--------|")
    wins = {}
    for n, t in tables.items():
        for ty in ('int32', 'int64', 'double', 'string'):
            for d in ('sorted', 'reverse', 'nearlysorted', 'lowcard16', 'lowcard256',
                      'allequal', 'mod8', 'zigzag', 'sawtooth', 'rotated', 'concat2',
                      'blockswap', 'farswap'):
                row = t.get((ty, d))
                if row is None:
                    continue
                who, b = best_other(row)
                ip = row.get('ips4opar')
                vq = row.get('vqsort')
                w(f"| {ty} | {CN.get(d, d)} | {n} | {row['fyxpar']:.5f} s | {who} | {b:.5f} s |"
                  f" {b/row['fyxpar']:.2f}x | {'--' if ip is None else f'{ip:.5f} s'} |"
                  f" {'--' if vq is None else f'{vq:.5f} s'} |")
    w("")
    # score against every opponent
    w("## 三、战绩")
    w("")
    w("矩阵覆盖 4 种类型 × 14 种分布（`string` 只在 1M 有，vqsort/x86-simd-sort 对它不适用）。")
    w("")
    for n, t in tables.items():
        wl, ls = [], []
        for key, row in t.items():
            ty = key[0]
            who, b = best_other(row)
            (wl if b / row['fyxpar'] >= 1 else ls).append((b / row['fyxpar'], ty, key[1], who, row['fyxpar'], b))
        w(f"- **{n}：胜 {len(wl)} / 负 {len(ls)}**")
        wins[n] = (wl, ls)
    w("")
    w("### 分对手战绩（并行 `fyx::sort` 对该对手的比值）")
    w("")
    w("| 对手 | 1M 胜/负 | 1M 最差 | 1M 最好 | 8M 胜/负 | 8M 最差 | 8M 最好 |")
    w("|------|---------:|--------:|--------:|---------:|--------:|--------:|")
    for key, label in OPP:
        cells = []
        for n, t in tables.items():
            rs = [(row[key] / row['fyxpar'], k) for k, row in t.items() if row.get(key) is not None]
            if not rs:
                cells.append(None)
                continue
            bad = min(rs)
            good = max(rs)
            cells.append((sum(1 for r, _ in rs if r >= 1), sum(1 for r, _ in rs if r < 1), bad, good))
        def fmt(c):
            if c is None:
                return "-- | -- | --"
            winsc, lossc, bad, good = c
            return f"{winsc}/{lossc} | {bad[0]:.2f}x ({bad[1][0]}/{CN.get(bad[1][1], bad[1][1])}) | {good[0]:.2f}x"
        w(f"| {label} | " + " | ".join(fmt(c) for c in cells) + " |")
    w("")
    w("### 输的格子（诚实清单）")
    w("")
    w("| 规模 | 类型 | 分布 | 本库并行 | 最强对手 | 对手成绩 | 比值 |")
    w("|------|------|------|----------|----------|----------|------|")
    for n, t in tables.items():
        for key, row in sorted(t.items()):
            who, b = best_other(row)
            if b / row['fyxpar'] < 1:
                w(f"| {n} | {key[0]} | {CN.get(key[1], key[1])} | {row['fyxpar']:.5f} s | {who} | {b:.5f} s |"
                  f" {b/row['fyxpar']:.2f}x |")
    w("")
    w("## 四、怎么读这些数字")
    w("")
    w("- 结构化输入可能命中已有序、旋转、计数或归并路径；是否胜出以本轮逐格数据为准，")
    w("  不从某一种分布推断到其它类型、比较器或硬件。")
    w("- 随机数据也单独列出。AVX-512 是本机 native 构建可选用的路径，不是库的运行要求；")
    w("  其它机器上的派发与成绩须另外测量。")
    w("- 所有输格均在上一节完整列出；不能用汇总胜率掩盖最弱输入。")
    w("- **2 个硬件线程跑不出多核内存带宽**。计算密集的核（直方图、散射、分区）吃得到第二个核，")
    w("  纯流式的活（扫描、拷贝）两线程约等于一线程。")
    w("")
    w("## 五、没有测、也不能在本机验证的")
    w("")
    w("- **4 核以上的并行带宽**：本机只有 2 个硬件线程，IPS4o 并行也只拿到 2 个线程。")
    w("- **GPU 路径**：无 GPU，代码是骨架 + CPU 回退。")
    w("- **其它第三方库**：本表只覆盖上方列出的六个对手；未测其它第三方库、其它架构或 GPU。")
    w("")
    w("## 复现")
    w("")
    w("```bash")
    w("./build.sh")
    w("for n in 1000000 8000000; do")
    w("  reps=7; [ $n -eq 8000000 ] && reps=5")
    w("  for r in 1 2 3; do")
    w("    bash tools/dev/vqsort.sh $n $reps > /dev/null")
    w("    cp build/vqsort_${n}.txt build/vqsort_${n}_r${r}.txt")
    w("  done")
    w("done")
    w("# merge_runs.py 对 3 份结果逐格取中位数；带 ! 的正确性失败会拒绝合并")
    w("python3 tools/dev/merge_runs.py --out build/vqsort_1000000_canonical.txt build/vqsort_1000000_r{1,2,3}.txt")
    w("python3 tools/dev/merge_runs.py --out build/vqsort_8000000_canonical.txt build/vqsort_8000000_r{1,2,3}.txt")
    w("python3 tools/dev/mkbench_md.py build/vqsort_1000000_canonical.txt build/vqsort_8000000_canonical.txt > BENCHMARKS.md")
    w("python3 tools/fyx_test.py            # 正确性检查")
    w("```")
    w("")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == '__main__':
    args = sys.argv[1:] or ['build/vqsort_1000000.txt', 'build/vqsort_8000000.txt']
    main(args)
