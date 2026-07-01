#!/bin/bash
# W7 step 2 prep — fetch SSSP-efficiency PBE pseudopotentials on the LOGIN node
# (Vanda/Atlas COMPUTE nodes are OFFLINE; download here). Result: finetune/pseudo/*.UPF,
# which 11_label_qe.py auto-discovers per element. Optionally pseudo/sssp.json for auto cutoffs.
#
# The SSSP set has no single stable hard-coded URL, so grab the 'SSSP efficiency (PBE)'
# download link from  https://www.materialscloud.org/discover/sssp/table/efficiency
# and pass it in:   SSSP_TAR_URL='https://.../SSSP_x.y_PBE_efficiency.tar.gz' bash 00_fetch_pseudos.sh
set -euo pipefail
cd "$(dirname "$0")"
PDIR="pseudo"; mkdir -p "$PDIR"

have_all() { for el in Li P S Cl; do
    ls "$PDIR"/${el}*.UPF "$PDIR"/${el}*.upf "$PDIR"/$(echo "$el" | tr A-Z a-z)*.UPF \
       "$PDIR"/$(echo "$el" | tr A-Z a-z)*.upf >/dev/null 2>&1 || return 1
  done; }

if have_all; then echo "[00] pseudo/ already has Li P S Cl UPFs -- nothing to do"; ls "$PDIR"; exit 0; fi

if [ -n "${SSSP_TAR_URL:-}" ]; then
    echo "[00] downloading SSSP from \$SSSP_TAR_URL ..."
    curl -fL "$SSSP_TAR_URL" -o /tmp/sssp_pbe_eff.tar.gz
    tar xzf /tmp/sssp_pbe_eff.tar.gz -C "$PDIR"
    # promote UPFs + the cutoff JSON to the top of pseudo/ (tarballs sometimes nest them)
    find "$PDIR" -type f -iname '*.upf'  -exec sh -c 'mv -f "$1" "$(dirname "$0")/$(basename "$1")"' "$PDIR" {} \; 2>/dev/null || true
    j=$(find "$PDIR" -type f -iname '*.json' | head -1 || true)
    [ -n "$j" ] && cp -f "$j" "$PDIR/sssp.json" && echo "[00] cutoffs -> pseudo/sssp.json"
elif command -v aiida-pseudo >/dev/null 2>&1; then
    echo "[00] no SSSP_TAR_URL; trying 'aiida-pseudo install sssp -x PBE -p efficiency' ..."
    aiida-pseudo install sssp -x PBE -p efficiency || true
    echo "[00] (export the family UPFs into finetune/pseudo/ per aiida-pseudo docs)"
fi

if have_all; then
    echo "[00] OK -> pseudo/ :"; ls "$PDIR"/*.UPF "$PDIR"/*.upf 2>/dev/null
else
    cat >&2 <<'EOF'
[00] pseudopotentials still MISSING. Do one of:
  (a) SSSP_TAR_URL='https://archive.materialscloud.org/.../SSSP_..._PBE_efficiency.tar.gz' bash 00_fetch_pseudos.sh
      (copy the 'SSSP efficiency (PBE)' file URL from materialscloud.org/discover/sssp/table/efficiency)
  (b) manually drop SSSP-efficiency PBE UPFs for Li P S Cl into finetune/pseudo/
Then re-run this script (idempotent) or go straight to 11_label_qe.py (it auto-discovers them).
EOF
    exit 1
fi
