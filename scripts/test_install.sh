#!/usr/bin/env bash
# --- phase4 additions ---------------------------------------------------------
# End-to-end install verification:
#   1. cmake --install the existing build tree into a temp prefix;
#   2. configure + build examples/consumer against it via find_package(umapcpp)
#      and run both smoke binaries (C++ and C API);
#   3. run the installed CLI: CSV fit-transform, npy round trip, and the
#      fit --save-model / transform --model split.
# Usage: scripts/test_install.sh [build-dir]   (default: ./build)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"

if [[ ! -d "$BUILD" ]]; then
    echo "test_install: build dir '$BUILD' not found; run cmake -B build && cmake --build build first" >&2
    exit 1
fi

WORK="$(mktemp -d /tmp/umapcpp-install-test.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
PREFIX="$WORK/prefix"

echo "== cmake --install -> $PREFIX"
cmake --install "$BUILD" --prefix "$PREFIX" > "$WORK/install.log"

fail=0
for f in \
    lib/libumapcpp.a \
    lib/libqhull_r.a \
    bin/umap \
    include/umap/umap.hpp \
    include/umap/umap_c.h \
    include/umap/metal_knn.h \
    lib/cmake/umapcpp/umapcpp-config.cmake \
    lib/cmake/umapcpp/umapcpp-config-version.cmake \
    lib/cmake/umapcpp/umapcpp-targets.cmake; do
    if [[ ! -e "$PREFIX/$f" ]]; then
        echo "MISSING from install tree: $f" >&2
        fail=1
    fi
done
# Metal builds also install the shaders (CPU-only builds legitimately don't).
if grep -q "UMAP_ENABLE_METAL:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null; then
    if [[ ! -e "$PREFIX/lib/umapcpp/umap_kernels.metallib" ]]; then
        echo "MISSING from install tree: lib/umapcpp/umap_kernels.metallib" >&2
        fail=1
    fi
fi
[[ $fail -eq 0 ]] || exit 1
echo "   install tree OK"

echo "== consumer project (find_package(umapcpp))"
CONS="$WORK/consumer-build"
cmake -S "$ROOT/examples/consumer" -B "$CONS" \
    -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
    > "$WORK/consumer-configure.log"
cmake --build "$CONS" -j4 > "$WORK/consumer-build.log"
"$CONS/consumer_smoke"
"$CONS/consumer_smoke_c"
echo "   consumer OK"

echo "== installed CLI"
DATA="$WORK/data"
mkdir -p "$DATA"
# 100 points: two 5-d blobs, deterministic (awk LCG), with a header row.
awk 'BEGIN {
    print "f0,f1,f2,f3,f4";
    seed = 20260828;
    for (i = 0; i < 100; i++) {
        shift = (i < 50) ? 0 : 10;
        line = "";
        for (d = 0; d < 5; d++) {
            seed = (seed * 1103515245 + 12345) % 2147483648;
            v = shift + seed / 2147483648.0;
            line = line sprintf("%.6f", v);
            if (d < 4) line = line ",";
        }
        print line;
    }
}' > "$DATA/blobs.csv"

UMAP_BIN="$PREFIX/bin/umap"

# fit-transform: CSV in, CSV out (header autodetected).
"$UMAP_BIN" fit-transform --input "$DATA/blobs.csv" --output "$DATA/emb.csv" \
    --n-neighbors 10 --init random --seed 42 --n-epochs 100
rows=$(wc -l < "$DATA/emb.csv" | tr -d ' ')
cols=$(head -1 "$DATA/emb.csv" | awk -F, '{print NF}')
if [[ "$rows" != 100 || "$cols" != 2 ]]; then
    echo "CLI fit-transform: expected 100x2 CSV, got ${rows}x${cols}" >&2
    exit 1
fi

# fit --save-model, then transform --model (no refit), npy output.
"$UMAP_BIN" fit --input "$DATA/blobs.csv" --save-model "$DATA/model.umap" \
    --n-neighbors 10 --init random --seed 42 --n-epochs 100
head -c 8 "$DATA/model.umap" | grep -q "UMAPCPP1" || {
    echo "CLI: model file missing UMAPCPP1 magic" >&2; exit 1; }
tail -n 20 "$DATA/blobs.csv" > "$DATA/queries.csv"
"$UMAP_BIN" transform --model "$DATA/model.umap" --input "$DATA/queries.csv" \
    --output "$DATA/qemb.npy"
if command -v python3 > /dev/null 2>&1; then
    python3 - "$DATA/qemb.npy" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    assert f.read(6) == b'\x93NUMPY', 'bad npy magic'
    major, minor = f.read(2)
    hlen = struct.unpack('<H', f.read(2))[0]
    header = f.read(hlen).decode()
    assert "'descr': '<f4'" in header and '(20, 2)' in header, header
    vals = struct.unpack('<40f', f.read(160))
    assert all(v == v for v in vals), 'NaN in transform output'
print('   CLI npy output OK (20x2 float32)')
PY
else
    # No python3: verify the magic + rough size instead.
    size=$(wc -c < "$DATA/qemb.npy" | tr -d ' ')
    head -c 6 "$DATA/qemb.npy" | grep -q "NUMPY" || {
        echo "CLI: qemb.npy is not an npy file" >&2; exit 1; }
    [[ "$size" -ge 224 ]] || { echo "CLI: qemb.npy too small" >&2; exit 1; }
    echo "   CLI npy output OK (magic + size)"
fi
echo "   CLI OK"

echo "test_install: ALL GREEN (prefix was $PREFIX)"
