#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Assembles the single-file header fyx_sort.hpp from parts/.
#
# The generic sorting-network templates live in parts/_network_body.inc and are
# expanded once per ISA region (marked //<<<NETWORK_BODY>>>).  They MUST be
# textually inside the region: GCC/Clang compile a template with the target
# attributes in effect at its point of *definition*, so a shared definition
# outside the region would be built for the baseline ISA and either fail to
# inline or change the vector ABI.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
out=fyx_sort.hpp
tmp=$(mktemp)
for f in $(ls parts/*.hpp | sort); do
  python3 - "$f" parts/_network_body.inc >> "$tmp" <<'PY'
import sys
src, body = sys.argv[1], sys.argv[2]
text = open(src).read()
b = open(body).read()
sys.stdout.write(text.replace('//<<<NETWORK_BODY>>>\n', b))
PY
  echo "" >> "$tmp"
done
echo "#endif // FYX_SORT_HPP_INCLUDED" >> "$tmp"
mv "$tmp" "$out"
chmod 644 "$out"
echo "built $out ($(wc -l < "$out") lines)"
