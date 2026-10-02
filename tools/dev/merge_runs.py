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
    names = None
    for line in open(path):
        if line.startswith("#") or line.startswith("(!"):
            footer.append(line.rstrip("\n")); continue
        if line.startswith("| type"):
            header.append(line.rstrip("\n"))
            cells = [x.strip() for x in line.rstrip("\n").split("|")]
            names = cells[4].split()
            continue
        if line.startswith("|---"):
            header.append(line.rstrip("\n")); continue
        if not line.startswith("| "):
            continue
        p = [x.strip() for x in line.rstrip("\n").split("|")]
        if len(p) < 7 or names is None:
            continue
        vals = p[4].split()
        if len(vals) != len(names):
            continue
        if any(value.endswith("!") for value in vals):
            raise ValueError(f"{path}: correctness verification failed in row {p[1:4]}")
        rows[(p[1], p[2], p[3])] = {
            "vals": {name: (None if value == "n/a" else float(value))
                     for name, value in zip(names, vals)}
        }
    return header, footer, names, rows

def fmt_row(t, n, d, vals, names, best_name, r_par, r_ser):
    cells = "   ".join("n/a" if vals[name] is None else f"{vals[name]:.6f}" for name in names)
    return (f"| {t:<7} | {n:>8} | {d:<13} |   {cells} "
            f"| {best_name:<12} {r_par:7.2f}x {r_ser:7.2f}x |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    headers, footers, names_by_run, tables = [], [], [], []
    for r in a.runs:
        try:
            h, f, names, t = parse(r)
        except ValueError as exc:
            ap.error(str(exc))
        headers.append(h); footers.append(f); names_by_run.append(names); tables.append(t)
    names = names_by_run[0]
    if not names or any(run_names != names for run_names in names_by_run[1:]):
        ap.error("all input runs must have the same timing-column header")
    if "fyx" not in names or "fyxpar" not in names:
        ap.error("input table must contain fyx and fyxpar timing columns")
    # Keep only cells present in every independent run, in the first-run order.
    ordered = [k for k in tables[0] if all(k in t for t in tables[1:])]
    lines = headers[0][:]
    opponents = [name for name in names if name not in ("fyx", "fyxpar")]
    for (t, n, d) in ordered:
        med = {}
        for name in names:
            samples = [table[(t, n, d)]["vals"][name] for table in tables]
            valid = [value for value in samples if value is not None]
            med[name] = None if not valid else statistics.median(valid)
        fy_par, fy_ser = med["fyxpar"], med["fyx"]
        best_name, best = "-", None
        for name in opponents:
            value = med[name]
            if value is not None and (best is None or value < best):
                best, best_name = value, name
        r_par = best / fy_par if best is not None and fy_par else 0.0
        r_ser = best / fy_ser if best is not None and fy_ser else 0.0
        lines.append(fmt_row(t, n, d, med, names, best_name, r_par, r_ser))
    lines.extend(footers[0])
    lines.append(f"# merge_runs: median of {len(a.runs)} independent process runs")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"merged {len(a.runs)} runs -> {a.out} ({len(ordered)} cells)")

if __name__ == "__main__":
    main()
