#!/usr/bin/env bash
# Update fyx sources and this benchmark harness from GitHub.
#   FYX_REPO=owner/name  FYX_BRANCH=branch  ./update.sh
main() {
    set -euo pipefail
    cd "$(dirname "$0")"
    local repo="${FYX_REPO:-$(cat suite/REPO 2>/dev/null || echo CLRV-FYX/fyx_sort)}"
    local branch="${FYX_BRANCH:-$(cat suite/BRANCH 2>/dev/null || echo arena/01a0f1e6-fyx-sort)}"
    local tmp; tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    echo "== 更新：$repo @ $branch =="
    if command -v git >/dev/null 2>&1; then
        git clone -q --depth 1 --branch "$branch" "https://github.com/$repo.git" "$tmp/src"
    elif command -v curl >/dev/null 2>&1; then
        mkdir -p "$tmp/x"; curl -fsSL "https://codeload.github.com/$repo/tar.gz/refs/heads/$branch" | tar -xz -C "$tmp/x"
        mv "$tmp/x/"* "$tmp/src"
    elif command -v wget >/dev/null 2>&1; then
        mkdir -p "$tmp/x"; wget -qO- "https://codeload.github.com/$repo/tar.gz/refs/heads/$branch" | tar -xz -C "$tmp/x"
        mv "$tmp/x/"* "$tmp/src"
    else
        echo "需要 git、curl 或 wget 之一"; exit 1
    fi
    local s="$tmp/src"
    [ -f "$s/fyx_sort.hpp" ] && [ -d "$s/bench/suite" ] || { echo "下载内容不完整"; exit 1; }

    rm -rf fyx.new && mkdir -p fyx.new
    cp "$s/fyx_sort.hpp" fyx.new/
    cp -R "$s/test" fyx.new/
    cp "$s/LICENSE" "$s/README.md" fyx.new/ 2>/dev/null || true
    (cd "$s" && git rev-parse HEAD 2>/dev/null || echo unknown) > fyx.new/COMMIT
    rm -rf fyx && mv fyx.new fyx

    mkdir -p suite
    for d in core algos runner; do rm -rf "suite/$d"; cp -R "$s/bench/suite/$d" suite/; done
    cp "$s/bench/suite/deps.lock" "$s/bench/suite/fetch_deps.sh" suite/
    [ -f "$s/bench/suite/update.ps1" ] && cp "$s/bench/suite/update.ps1" suite/
    for f in start.sh start.cmd update.cmd README.md; do cp "$s/bench/suite/$f" .; done
    chmod +x start.sh suite/fetch_deps.sh
    rm -rf build   # force rebuild with the new sources (third_party is kept)

    if [ ! -f third_party/LOCK ] || ! cmp -s third_party/LOCK suite/deps.lock; then
        if command -v git >/dev/null 2>&1; then bash suite/fetch_deps.sh "$PWD/third_party"
        else echo "警告：依赖版本有变，但没有 git；start.sh 运行时会提示。"; fi
    fi
    cp "$s/bench/suite/update.sh" ./update.sh.new && chmod +x update.sh.new && mv update.sh.new update.sh
    echo "fyx 已更新到 $(cat fyx/COMMIT)。运行 ./start.sh 开始测评。"
}
main "$@"
exit
