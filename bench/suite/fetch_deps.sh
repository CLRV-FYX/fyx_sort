#!/usr/bin/env bash
# Fetch the pinned third-party sources listed in deps.lock into $1
# (default: ../third_party relative to this script).  Requires git.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${1:-$here/../third_party}"
cache="${FYX_DEPS_CACHE:-${TMPDIR:-/tmp}/fyx_deps_cache}"
mkdir -p "$dest" "$cache"
dest="$(cd "$dest" && pwd)"

fetch_ref() {   # fetch_ref <url> <ref> <dir>
    local url="$1" ref="$2" dir="$3"
    if [ ! -d "$dir/.git" ]; then git init -q "$dir"; git -C "$dir" remote add origin "$url"; fi
    # A cached checkout already pinned to this ref needs no network at all.
    if [ "$(cat "$dir/.git/fyx_pin" 2>/dev/null)" = "$ref" ] &&
       git -C "$dir" rev-parse -q --verify "refs/fyx/pin^{commit}" >/dev/null 2>&1; then
        git -C "$dir" -c advice.detachedHead=false checkout -q --force refs/fyx/pin
        return 0
    fi
    if git -C "$dir" fetch -q --depth 1 origin "$ref:refs/fyx/pin" 2>/dev/null ||
       git -C "$dir" fetch -q --depth 1 origin "+refs/tags/$ref:refs/fyx/pin" 2>/dev/null; then
        echo "$ref" > "$dir/.git/fyx_pin"
        git -C "$dir" -c advice.detachedHead=false checkout -q --force refs/fyx/pin
        return 0
    fi
    # Offline: accept an existing checkout of exactly this ref (SHA or tag).
    if git -C "$dir" rev-parse -q --verify "$ref^{commit}" >/dev/null 2>&1; then
        git -C "$dir" -c advice.detachedHead=false checkout -q --force "$ref"
        return 0
    fi
    if [ -n "${FYX_DEPS_ALLOW_STALE:-}" ] && git -C "$dir" rev-parse -q --verify HEAD >/dev/null 2>&1; then
        echo "  [deps] WARNING: cannot fetch $url @ $ref; using cached HEAD $(git -C "$dir" rev-parse --short HEAD)" >&2
        return 0
    fi
    echo "  [deps] ERROR: cannot fetch $url @ $ref" >&2
    return 1
}

while IFS='|' read -r name url ref paths; do
    case "$name" in ''|'#'*) continue ;; esac
    echo "  [deps] $name @ $ref"
    if [ "$name" = boost ]; then
        rm -rf "$dest/boost"; mkdir -p "$dest/boost"
        for m in $paths; do
            fetch_ref "https://github.com/boostorg/$m.git" "$ref" "$cache/boost_$m"
            cp -R "$cache/boost_$m/include/boost" "$dest/boost/"
        done
        cp "$cache/boost_sort/LICENSE_1_0.txt" "$dest/boost/" 2>/dev/null || true
        continue
    fi
    fetch_ref "$url" "$ref" "$cache/$name"
    rm -rf "$dest/$name"; mkdir -p "$dest/$name"
    for p in $paths; do
        if [ -e "$cache/$name/$p" ]; then
            mkdir -p "$dest/$name/$(dirname "$(basename "$p")")"
            cp -R "$cache/$name/$p" "$dest/$name/"
        fi
    done
done < "$here/deps.lock"
# Highway: tests / benchmarks are not needed.
find "$dest/highway" -name '*_test.cc' -delete 2>/dev/null || true
rm -rf "$dest/highway/hwy/tests" "$dest/highway/hwy/contrib/sort/bench_"*.cc 2>/dev/null || true
cp "$here/deps.lock" "$dest/LOCK"
echo "  [deps] done -> $dest"
