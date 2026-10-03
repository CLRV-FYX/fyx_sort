#!/bin/bash
# run_tests.sh TAG "FLAGS" [t_x ...]   (logs in /tmp/tb)
TAG=$1; FLAGS=$2; shift 2
mkdir -p /tmp/tb
cd "$(dirname "$0")/../../../test"
TS=${@:-$(ls t_*.cpp | sed 's/.cpp//')}
one() { t=$1; out=/tmp/tb/${t}_$TAG; if g++ -std=c++17 -O2 -Wall -Wextra -Werror $FLAGS -pthread $t.cpp -o $out 2>/tmp/tb/${t}_$TAG.err; then if timeout 900 $out > /tmp/tb/${t}_$TAG.log 2>&1; then echo "PASS $t"; else echo "FAIL(run) $t"; tail -5 /tmp/tb/${t}_$TAG.log; fi; else echo "FAIL(build) $t"; head -20 /tmp/tb/${t}_$TAG.err; fi; }
export -f one; export TAG FLAGS
printf "%s\n" $TS | xargs -P 2 -I{} bash -c 'one {}'
