#!/usr/bin/env bash
# Build the self-contained benchmark package (run from anywhere in the repo).
#   bash bench/suite/make_package.sh [output.tar.gz]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="${1:-$repo/packages/fyx_bench_suite.tar.gz}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
stage="$work/fyx_bench_suite"
mkdir -p "$stage/suite" "$stage/fyx"
cp "$here"/{start.sh,start.cmd,update.sh,update.cmd,README.md} "$stage/"
cp -R "$here"/{core,algos,runner} "$stage/suite/"
cp "$here"/{deps.lock,fetch_deps.sh,update.ps1,REPO,BRANCH} "$stage/suite/"
cp "$repo/fyx_sort.hpp" "$repo/LICENSE" "$repo/README.md" "$stage/fyx/"
cp -R "$repo/test" "$stage/fyx/"
(cd "$repo" && git rev-parse HEAD) > "$stage/fyx/COMMIT"
if [ -n "$(cd "$repo" && git status --porcelain -- fyx_sort.hpp test 2>/dev/null)" ]; then echo "+uncommitted" >> "$stage/fyx/COMMIT"; fi
bash "$here/fetch_deps.sh" "$stage/third_party"
chmod +x "$stage/start.sh" "$stage/update.sh" "$stage/suite/fetch_deps.sh"
mkdir -p "$(dirname "$out")"
tar -czf "$out" -C "$work" fyx_bench_suite
echo "package: $out ($(du -h "$out" | cut -f1))"
