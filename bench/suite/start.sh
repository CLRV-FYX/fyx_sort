#!/usr/bin/env bash
# fyx benchmark suite -- one-command entry point (Linux / macOS / WSL / MSYS2).
#
#   ./start.sh                     full profile with the primary compiler, then a
#                                  quick pass with every other detected compiler
#   ./start.sh --profile=quick     faster smoke run
#   ./start.sh --profile=huge      adds 1e8-element cases (needs >= 16 GB RAM)
#   FYX_ALL_COMPILERS=0 ./start.sh only the primary compiler
#   CXX=clang++ ./start.sh         choose the primary compiler
# Any extra arguments are passed to the runner (./start.sh --help for the list).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

echo "== fyx 基准测评：依赖检测 =="
have() { command -v "$1" >/dev/null 2>&1; }
compilers=()
for c in ${CXX:-} g++ clang++ c++; do
    [ -n "$c" ] || continue
    if have "$c"; then
        real="$(command -v "$c")"
        dup=0
        for x in "${compilers[@]:-}"; do [ "$x" = "$c" ] && dup=1; done
        # skip c++ when it is just an alias of an already listed compiler
        if [ "$c" = c++ ] && [ ${#compilers[@]} -gt 0 ]; then
            for x in "${compilers[@]}"; do [ "$(readlink -f "$(command -v "$x")" 2>/dev/null || true)" = "$(readlink -f "$real" 2>/dev/null || true)" ] && dup=1; done
        fi
        [ $dup = 0 ] && compilers+=("$c")
    fi
done
if [ ${#compilers[@]} -eq 0 ]; then
    echo "未找到 C++17 编译器。请安装后重试："
    echo "  Debian/Ubuntu: sudo apt install build-essential   (可选 clang)"
    echo "  Fedora/RHEL:   sudo dnf install gcc-c++"
    echo "  Arch:          sudo pacman -S base-devel"
    echo "  macOS:         xcode-select --install"
    exit 1
fi
echo "  编译器: ${compilers[*]}"
have tar   && echo "  tar: 有" || echo "  tar: 无（结果需手动打包）"
have git   && echo "  git: 有" || echo "  git: 无（仅在 third_party 缺失或更新时需要）"
if have rustc; then echo "  rustc: $(rustc --version)"; else echo "  rustc: 无（Rust std sort 对手将标为不可用；安装 https://rustup.rs 可加入）"; fi
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "  CPU governor: $gov"
    [ "$gov" = performance ] || echo "  提示：若可行请切到 performance（sudo cpupower frequency-set -g performance），并关闭其它负载。"
fi

if [ ! -f third_party/LOCK ] || ! cmp -s third_party/LOCK suite/deps.lock; then
    if have git; then
        echo "== 获取/更新第三方依赖（固定版本）=="
        bash suite/fetch_deps.sh "$ROOT/third_party"
    else
        echo "警告：third_party 缺失或与 deps.lock 不一致，且没有 git；缺失的对手会被标为不可用。"
    fi
fi

mkdir -p build
primary="${compilers[0]}"
echo "== 构建 runner（$primary）=="
if ! "$primary" -std=c++17 -O2 suite/runner/runner.cpp -o build/runner -pthread 2>build/runner_build.log; then
    "$primary" -std=c++17 -O2 suite/runner/runner.cpp -o build/runner -pthread -lstdc++fs 2>>build/runner_build.log || {
        cat build/runner_build.log; echo "runner 构建失败"; exit 1; }
fi

session=()
run_one() {   # run_one <cxx> [args...]
    local cxx="$1"; shift
    local cc=""
    case "$cxx" in *clang++*) cc="${cxx%++}";; *g++*) cc="${cxx%g++}gcc";; esac
    ./build/runner --root="$ROOT" --cxx="$cxx" ${cc:+--cc="$cc"} --no-pack "$@" || echo "runner($cxx) 返回非零，继续。"
    [ -f results/LATEST ] && session+=("$(cat results/LATEST)")
}

echo "== 主测评：$primary =="
run_one "$primary" "$@"
if [ "${FYX_ALL_COMPILERS:-1}" = 1 ] && [ ${#compilers[@]} -gt 1 ]; then
    for c in "${compilers[@]:1}"; do
        echo "== 附加编译器快速测评：$c =="
        run_one "$c" --profile=quick --skip-tests "$@"
    done
fi

if have tar && [ ${#session[@]} -gt 0 ]; then
    out="fyx_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$out" -C results "${session[@]}"
    echo
    echo "=================================================================="
    echo " 完成。请把这个文件发回： $ROOT/$out"
    echo " 各次报告： ${session[*]/#/results/}/report.md"
    echo "=================================================================="
fi
