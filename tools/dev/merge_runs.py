#!/usr/bin/env python3
"""Merge several bench_matrix runs into one canonical table, per-cell median.

Every cell's eight timing columns are medianed component-wise across the
input runs (n/a stays n/a only when every run has it), and the best-other
columns are recomputed from the medianed values, so the merged table keeps
the exact bench_matrix format and can be fed straight into mkbench_md.py.

    python3 tools/dev/merge_runs.py --out build/vqsort_1000000.txt run1 run2 run3
"""
import argparse
import statistics
import sys

def parse(path):
    header, footer, rows = [], [], {}
    for line in open(path):
        if line.startswith("#") or line.startswith("(!"):
            footer.append(line.rstrip("\n")); continue
        if line.startswith("| type"):
            header.append(line.rstrip("\n")); continue
        if line.startswith("|---"):
            header.append(line.rstrip("\n")); continue
        if not line.startswith("| "):
            continue
        p = [x.strip() for x in line.rstrip("\n").split("|")]
        if len(p) < 7:
            continue
        vals = p[4].split()
        tail = p[5].split()
        if len(vals) < 8 or len(tail) < 3:
            continue
        rows[(p[1], p[2], p[3])] = {
            "vals": [None if v == "n/a" else float(v) for v in vals],
            "tail_name": tail[0],
            "tail_gain": tail[2],
        }
        # n rides inside the key already
    return header, footer, rows

def fmt_row(t, n, d, vals, best_name, r_par, r_ser):
    cells = "   ".join("n/a" if v is None else f"{v:.6f}" for v in vals)
    return (f"| {t:<7} | {n:>8} | {d:<13} |   {cells} "
            f"| {best_name:<12} {r_par:7.2f}x {r_ser:7.2f}x |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    headers, footers, tables = [], [], []
    for r in a.runs:
        h, f, t = parse(r)
        headers.append(h); footers.append(f); tables.append(t)
    # keys are (type, dist); keep first-seen ordering of run 1
    ordered = [k for k in tables[0] if all(k in t for t in tables[1:])]
    lines = headers[0][:]
    for (t, n, d) in ordered:
        cols = list(zip(*[tt[(t, n, d)]["vals"] for tt in tables]))
        med = [None if all(v is None for v in c) else statistics.median([v for v in c if v is not None]) for c in cols]
        fy_par, fy_ser = med[1], med[0]
        best_name, best = "-", None
        names = ["std", "ips4o", "ips4opar", "pdqsort", "vqsort", "xss"]
        for i, nm in zip(range(2, 8), names):
            v = med[i]
            if v is None: continue
            if best is None or v < best: best, best_name = v, nm
        r_par = best / fy_par if fy_par and best else 0.0
        r_ser = best / fy_ser if fy_ser and best else 0.0
        lines.append(fmt_row(t, n, d, med, best_name, r_par, r_ser))
    lines.extend(footers[0])
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"merged {len(a.runs)} runs -> {a.out} ({len(ordered)} cells)")

if __name__ == "__main__":
    main()
