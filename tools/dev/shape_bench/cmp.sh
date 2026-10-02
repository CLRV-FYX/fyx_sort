#!/bin/bash
# ./cmp.sh TAG [n...]  : fyx vs xss, ratio = xss/fyx (<1 = fyx loses)
TAG=${1:-cur}; shift; NS=${@:-1000 20000 200000}
for ty in int32_t double; do for n in $NS; do echo "== $ty $n"
  paste <(/tmp/sh_${TAG}_$ty $n) <(/tmp/shx_$ty $n) | awk '{r=$4/$2; printf "%-14s fyx=%s xss=%s ratio=%.2f %s\n",$1,$2,$4,r,(r<1?"LOSE":"")}'
done; done
