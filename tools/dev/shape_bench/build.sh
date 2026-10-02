#!/bin/bash
# ./build.sh HDR_PATH TAG [extra flags]  -> /tmp/sh_TAG_{int32_t,double}; xss once.
# Compiles at most 2 at a time (4 GB sandbox).
set -e
cd "$(dirname "$0")"
HDR=${1:-../../../fyx_sort.hpp}; TAG=${2:-cur}; shift 2 || true; XSS_DIR=${XSS_DIR:-/tmp/xss}
for ty in int32_t double; do
  g++ -std=c++17 -O2 -march=native "$@" -DHDR="\"$(realpath $HDR)\"" -DTY=$ty sh.cpp -o /tmp/sh_${TAG}_$ty &
done; wait
if [ ! -x /tmp/shx_double ] && [ -d "$XSS_DIR/src" ]; then
  for ty in int32_t double; do g++ -std=c++17 -O2 -march=native -DXSS -I$XSS_DIR/src -DTY=$ty sh.cpp -o /tmp/shx_$ty & done; wait
fi
