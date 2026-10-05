#!/bin/bash
# Rebuild the micro-benchmarks in /tmp/w (lost on sandbox resets).
# Needs: bash bench/suite/fetch_deps.sh /tmp/w/tp
set -e
R=$(cd "$(dirname "$0")/../../../.." && pwd)
W=/tmp/w; mkdir -p $W; cd $W
S=$R/bench/suite/algos/scandum.c
for k in 0:crum:crumsort 1:flux:fluxsort; do
  IFS=: read id nm dir <<<"$k"
  gcc -O3 -march=native -c -DFB_SCANDUM=$id -I$W/tp/$dir $S -o ${nm}_raw.o
  args=""; for t in i32 u32 i64 u64 f32 f64 rec; do args="$args --redefine-sym fb_scandum_$t=${nm}_$t"; done
  : > ${nm}.keep; for t in i32 u32 i64 u64 f32 f64 rec; do echo ${nm}_$t >> ${nm}.keep; done
  objcopy $args ${nm}_raw.o ${nm}_ren.o
  objcopy --keep-global-symbols=${nm}.keep ${nm}_ren.o ${nm}2.o   # helpers local: no clash
done
g++ -std=c++17 -O3 -DNDEBUG -march=native -I$R -I$R/bench/suite $R/tools/dev/round/mb/so.cpp crum2.o flux2.o -o so -pthread
echo built $W/so
