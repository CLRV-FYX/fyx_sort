#!/bin/bash
# ./confirm.sh TAG TYPE N DIST [reps]: alternate fyx / xss runs of one shape,
# each its own process, report min of each (robust against drift/noise).
TAG=$1; TY=$2; N=$3; D=$4; R=${5:-5}; a=1e9; b=1e9
for i in $(seq $R); do
  x=$(/tmp/sh_${TAG}_$TY $N $D | awk '{print $2}'); y=$(/tmp/shx_$TY $N $D | awk '{print $2}')
  a=$(awk -v a=$a -v x=$x 'BEGIN{print (x<a)?x:a}'); b=$(awk -v b=$b -v y=$y 'BEGIN{print (y<b)?y:b}')
done
awk -v a=$a -v b=$b -v t=$TY -v n=$N -v d=$D 'BEGIN{printf "%s n=%s dist=%s fyx=%.3f xss=%.3f ratio=%.2f %s\n",t,n,d,a,b,b/a,(b<a?"LOSE":"")}'
