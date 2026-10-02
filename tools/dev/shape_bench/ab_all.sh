#!/bin/bash
# ./ab_all.sh HDR [n...]: in-process alternating A/B (fyx vs xss), median ns/elem, all shapes.
HDR=$(realpath ${1:-../../../fyx_sort.hpp}); shift; NS=${@:-1000 20000 200000}
D=$(dirname $0)
for ty in int32_t double; do g++ -std=c++17 -O2 -march=native -I${XSS:-/tmp/xss/src} -DHDR="\"$HDR\"" -DTY=$ty $D/ab.cpp -o /tmp/aball_$ty & done; wait
for ty in int32_t double; do for n in $NS; do echo "== $ty $n"; for d in $(seq 0 15); do /tmp/aball_$ty $n $d; done; done; done
