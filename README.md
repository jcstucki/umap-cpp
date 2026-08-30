# UMAP-C++

A complete, faithful C++17 reproduction of [UMAP](https://github.com/lmcinnes/umap)
(`umap-learn`), GPU-parallelizable via Apple Metal, consumable as a static
library (C++ and C APIs) and a Unix CLI executable.

"Faithful" here means the same algorithms, constants, defaults, quirks and —
wherever the arithmetic allows — the same bits as the Python reference
(`umap-learn` + `pynndescent`, serial `NUMBA_NUM_THREADS=1` path), not merely
"an implementation of the UMAP paper".

## Faithfulness

The port replicates the reference implementation down to numba float32/float64
typing, numpy legacy `RandomState` streams (MT19937, draw for draw), scipy CSR
buffer-sharing quirks, sklearn pairwise-distance kernels, and vendored
byte-identical qhull sources for `inverse_transform`'s Delaunay
triangulation. On seeded configurations the following are **bit-exact** against
the Python reference (validated by golden tests generated from the real
implementation): small-data fits and transforms (including supervised,
densMAP + local radii, metric/output-metric kwds, precomputed metrics and
kNN, `unique=True`, ndarray/PCA init stages, n_epochs lists, graph mode,
`inverse_transform`, chained `update()`s, composite `*`/`+`/`-` graphs), the
whole kNN/fuzzy-graph pipeline of the NN-descent branch (indices identical;
distances ~1 ulp due to numba fastmath), and sparse-input equivalents.
Stochastic-quality paths are validated with the upstream test-suite thresholds
(trustworthiness / kNN accuracy / ARI, ported verbatim). The hardening pass
extends the surface to upstream's full sparse small-data metric routing
(including the float64 sklearn manhattan kernel and the densify fallback),
sparse precomputed distance matrices, sparse search-index persistence,
callable metrics (`custom_metric`/`custom_metric_grad`/`custom_output_metric`
as the `std::function` equivalents of upstream's callables), and `n_jobs`
parallelism gating (seeded fits force `n_jobs=1` with upstream's warning).

An adversarial differential audit (2026-08-29, recorded in
[docs/AUDIT-2026-08-29.md](docs/AUDIT-2026-08-29.md)) re-proved the bit-exact
surface across 21 cases + 17 fuzz edges and found four real deviations, all
since fixed and golden-locked: `update()` with default `n_epochs` (upstream
runs ZERO SGD epochs), the supervised `target_metric="count"` float32 value
chain, transform-side exact-tie neighbour selection (numpy's introselect /
introsort now ported verbatim next to numba's fit-side quickselect), and two
degenerate corners (the n=2 empty-graph `ValueError`, and
`ensure_all_finite="allow-nan"`'s NaN pairwise diagonal + NaN-propagating
`np.max` pipeline).

Documented deviations (each noted at its definition site and in
[ROADMAP.md](ROADMAP.md)): Spectra-vs-ARPACK eigensolver residuals in spectral
init (~1e-6, up to sign), `find_ab_params` reaching scipy's optimum to ~1e-7,
tau-prng standing in for `RandomState` in some multi-component spectral
branches, exact-tie ordering in the two full-row `np.argsort` sites
(`update()`'s init-neighbour selection and `inverse_transform`'s candidate
ordering; every k-selection site is now tie-exact), sparse `init="pca"`
(sklearn's randomized TruncatedSVD is not bit-replicable) and callable
metrics on sparse input (the CSR callable signature) both raising, kNN
construction staying single-threaded regardless of `n_jobs`
(performance-only; bit-exact with `NUMBA_NUM_THREADS=1`), `"custom"`-metric
models not being serialisable, and the GPU path's float32/hogwild semantics
(below). `ROADMAP.md` is the authoritative, per-feature log of parity levels
and deviations.

## Building

Requirements: CMake ≥ 3.16, a C++17 compiler. Eigen 3.4 and Spectra 1.0.1 are
fetched automatically; metal-cpp and qhull are vendored. On macOS the Metal
GPU kernels build by default (Xcode command-line tools provide `xcrun metal`).

```sh
cmake -B build && cmake --build build -j8
./build/umap_tests          # doctest suite
./build/umap --help         # CLI

# CPU-only build (also the configuration for non-Apple platforms):
cmake -B build-nometal -DUMAP_ENABLE_METAL=OFF
cmake --build build-nometal -j8
```

Install + package config:

```sh
cmake --install build --prefix /usr/local   # or any prefix
scripts/test_install.sh                     # end-to-end install verification
```

This installs `lib/libumapcpp.a` + `lib/libqhull_r.a`, `bin/umap`, the headers
under `include/umap/`, a CMake package (`find_package(umapcpp)`), and — on
Metal builds — `lib/umapcpp/umap_kernels.metallib`. The Metal shader library
is located at runtime via `$UMAP_METALLIB_PATH`, the build-tree path, or
relative to the executable (`<bindir>/../lib/umapcpp/`).

## CLI

```sh
# Fit and write the embedding (CSV in — header autodetected — CSV out):
umap fit-transform --input data.csv --output embedding.csv \
    --n-neighbors 15 --min-dist 0.1 --metric euclidean --seed 42

# NumPy arrays work directly (.npy float32/float64, C order, 1-D/2-D):
umap fit-transform --input data.npy --output embedding.npy --densmap

# Fit once, save the model (params + data + graph + search index + RNG
# bookkeeping in a versioned binary format), then transform new points
# without refitting:
umap fit --input train.npy --save-model model.umap --seed 42
umap transform --model model.umap --input new_points.npy --output emb.npy

# Supervised fit, Minkowski metric with p=3, GPU layout:
umap fit-transform --input X.csv --y labels.csv --output emb.csv \
    --metric minkowski --metric-kwds p=3 --device gpu
```

All main hyperparameters are exposed (`umap --help`): n-neighbors,
n-components, metric (+ `--metric-kwds p=… z=… v=a:b:…`), min-dist, spread,
n-epochs, learning-rate, init (`spectral|random|pca|tswspectral` or a matrix
file), random-state/seed, negative-sample-rate, repulsion-strength,
set-op-mix-ratio, local-connectivity, a/b, densmap (+ dens-lambda/frac/
var-shift, output-dens), output-metric, supervised `--y` (+ target-metric/
weight/n-neighbors), disconnection-distance, unique, force-approximation,
device cpu|gpu, verbose.

## Library usage

CMake:

```cmake
find_package(umapcpp 1.0 REQUIRED)
target_link_libraries(your_target PRIVATE umapcpp::umapcpp)
```

C++ (`include/umap/umap.hpp` mirrors `umap.UMAP`):

```cpp
#include "umap/umap.hpp"

umap::UMAPParams params;
params.n_neighbors = 15;
params.min_dist = 0.1;
params.random_state = 42;          // numpy RandomState stream parity
umap::UMAP model(params);

std::vector<float> emb = model.fit_transform(X.data(), n_samples, n_features);
std::vector<float> q   = model.transform(Xnew.data(), n_queries);

model.save("model.umap");                       // versioned binary format
umap::UMAP loaded = umap::UMAP::load("model.umap");
```

Sparse CSR input, supervised `fit(X, y)`, `update()`, `inverse_transform`,
`transform_mode="graph"`, precomputed metrics/kNN and composite models
(`a * b`, `a + b`, `a - b`) are all available on the same class — see the
header documentation.

C (`include/umap/umap_c.h`, links against the same archive):

```c
#include "umap/umap_c.h"

umap_params p;
umap_params_init(&p);
p.n_neighbors = 10;
p.has_random_state = 1; p.random_state = 42;

umap_handle* h = umap_create(&p);
float* emb = malloc(n * p.n_components * sizeof(float));
if (umap_fit_transform(h, X, n, dim, NULL, emb) != UMAP_OK)
    fprintf(stderr, "umap: %s\n", umap_last_error(h));
umap_destroy(h);
```

A complete `find_package` consumer (C++ and C) lives in
[`examples/consumer/`](examples/consumer/).

## Performance

Measured on an M-series Mac (median of the benchmark harness; `~/.venvs`
Python reference = numba 0.67, NUMBA_NUM_THREADS=1 for serial). Embedding
quality is validated in lockstep — trustworthiness on the seeded CPU path is
within ±0.0004 of the reference at every size, so the numbers below trade no
accuracy.

Spectral initialisation uses a **matvec-only Lanczos** on the flipped
normalised Laplacian `2I − L` (largest-algebraic eigenpairs of `2I − L` are
the smallest of `L`), mirroring upstream's `eigsh(which="SM", tol=1e-4)`
instead of a shift-invert whose sparse-LU factorisation dominated large fits.
At 50k×16 this cuts the spectral-init stage from **75.2 s → 0.33 s (~230×)**,
taking the whole seeded CPU fit from ~98.6 s to ~20.6 s (SGD-bound now) — at
parity with the Python reference's serial fit (21.9 s).

| fit, 50k×16, seeded | spectral init | whole fit |
| --- | ---: | ---: |
| before (shift-invert) | 75,203 ms | 98,594 ms |
| after (2I−L Lanczos)  | 327 ms | ~20,600 ms |
| Python (numba serial) | — | 21,863 ms |

`--device gpu` (below) then moves kNN and the SGD onto Metal (with the
register-blocked exact-kNN kernel — see the GPU section): the same 50k×16 fit
runs in **~1.03 s**, 50k×64 in **~1.67 s**, and 10k×64 in **~0.52 s** — 3–5×
faster than the Python reference's *parallel* numba fits. The deterministic
seeded CPU path is at parity with Python's serial fit; the unseeded CPU path
(upstream's own hogwild `parallel` semantics + the P2-GATED fast-`pow`) matches
Python's parallel fit on CPU alone. End-to-end fit wall-clock (M-series Mac):

| fit | seeded CPU | unseeded CPU | device=gpu | py serial / parallel |
| --- | ---: | ---: | ---: | ---: |
| 10k×64 | 10.5 s | 2.04 s | **0.52 s** | 11.0 / 2.44 s |
| 50k×16 | 20.7 s | 4.46 s | **1.03 s** | 21.9 / 4.66 s |
| 50k×64 | 24.7 s | 7.00 s | **1.67 s** | 23.9 / 5.20 s |

### SGD layout on the CPU

With spectral init reduced to ~0.3 s, the SGD layout is the dominant stage of a
CPU fit (~95% of a seeded fit; ~1.4–2.7 s of an unseeded one). Three changes
speed it up **without touching the seeded bit-contract** — the seeded/sequential
path (`random_state` set) still calls libm `pow` in the same order as numba, so
its layout goldens stay byte-identical (max-abs-diff 0):

* a compile-time `dim==2` / non-densMAP specialization of the epoch kernel
  (unrolled coordinate loops, density branches elided, dead edges skipped) —
  control-flow only, so it is bit-exact on every path. Seeded SGD **−2%** (the
  seeded path is libm-`pow`-bound — ~2/3 of its time is `pow`, which the numba
  parity pins);
* a **persistent hogwild thread pool** replacing per-epoch `std::thread`
  spawns (a 500-epoch fit spawned thousands of threads). Same static work
  partition, same race semantics; unseeded SGD **−0.5 to −4%**, scaling with
  epoch count;
* a polynomial **fast `pow`** (`exp2(y·log2(x))`, ~3.8e-4 relative error) on the
  **unseeded hogwild path only** — that path is already non-deterministic
  (numba's `parallel=True` semantics), so it can trade the last ULPs of `pow`
  for speed while the seeded path stays exact. Unseeded SGD **−26 to −30%**.
  Trustworthiness over median-of-5 unseeded fits is unchanged vs the libm build
  (10k: +0.0001, within sampling noise).

Unseeded (hogwild-parallel) SGD stage, before (P0 baseline) → after, ms
(M-series, seeded uses libm `pow` throughout; the wins compound on the unseeded
path):

| SGD stage, ms | seeded (bit-exact) | unseeded before | unseeded after |
| --- | ---: | ---: | ---: |
| 10k×16 | 9,447 → 9,284 | 2,200 | **1,385** (−37%) |
| 10k×64 | 9,937 → 9,745 | 2,300 | **1,500** (−35%) |
| 50k×16 | ~19,200 | 3,600 | **2,510** (−30%) |
| 50k×64 | ~20,300 | 3,850 | **2,667** (−31%) |

The seeded stage is bounded by libm `pow` that the reference pins; the big lever
is the unseeded path, where an unseeded 50k×64 fit now runs in ~7.1 s (SGD
2.7 s) and 10k×64 in ~2.0 s. For the fastest seeded large fits, use
`--device gpu` (above).

## GPU (Apple Metal)

`UMAPParams::device = umap::Device::GPU` (CLI: `--device gpu`) routes to Metal
kernels where they genuinely cover the computation:

* **every kNN computation site** runs on a tiled exact brute-force engine
  (one thread per query, data streamed through threadgroup-memory tiles — any
  n, any query count; per-thread top-k registers for k ≤ 64, a
  (dist, index)-heap for 64 < k ≤ 256);
* the layout SGD of euclidean-output **fits — densMAP included — and of
  transform** (one thread per edge, hogwild float32 updates — the same race
  semantics as numba's `parallel=True`). Transform's `move_other=false` SGD
  never writes the frozen training rows (races confined to the query side);
  densMAP keeps its per-epoch reductions host-side with the exact CPU numpy
  float32 semantics (shared-storage embedding buffer) and runs the per-edge
  density correction on the GPU. 5k-point benchmarks vs the sequential CPU
  loop (hidden cases `-ns -tc="gpu sgd timing*"` / `-tc="gpu layout
  timing*"`): plain fit 191 ms vs 1724 ms (~9×), densMAP 280 ms vs 1913 ms
  (~6.8×, 200 epochs / 60 corrected), transform 84 ms vs 175 ms (~2×, 1000
  queries, 100 epochs).

kNN coverage by site (device=GPU; anything not covered falls back to the
bit-exact CPU port with a stderr note):

| kNN site | engine-covered metrics | otherwise |
| --- | --- | --- |
| dense fit, small-data branch (n < 4096) | euclidean/l2, sqeuclidean, manhattan/l1/taxicab, chebyshev/linf\*, minkowski(p), cosine, correlation, hamming, canberra, braycurtis, seuclidean(v), wminkowski(w, p), mahalanobis(Vinv) — k ≤ 256 | CPU pairwise |
| dense fit, standard branch (4096 ≤ n ≤ 100k) | same 13 metrics, k ≤ 256, `unique=False` — the engine's **exact** kNN graph replaces NN-descent's approximate one (recall 1.0; no search index is built) | CPU NN-descent |
| dense fit, standard branch (n > 100k) | euclidean/l2, sqeuclidean, manhattan, cosine, k ≤ 64, `unique=False` — **GPU NN-descent** (host RP-forest init + on-device descent; recall ≈ CPU NN-descent's to ~1e-4, see below); other engine metrics / 64 < k ≤ 256 stay on the exact GPU engine at any n | CPU NN-descent |
| dense transform (any n) | same 13 metrics, k ≤ 256 — exact engine kNN vs the training data, no search index needed | CPU `PreparedIndex` query |
| sparse (CSR) fit small + standard, sparse transform | euclidean, cosine (norm-trick CSR kernel) | CPU sparse kernels / NN-descent |
| precomputed / `precomputed_knn` | n/a (no distance computation) | — |
| `update()`, supervised target-y kNN, `output_dens` embedding density, `inverse_transform` ranking, composites | CPU only (RandomState-stream- or bit-exactness-critical; `update()` additionally needs the NN-descent index a GPU standard fit skips, so it raises) | CPU |

Discrete/object metrics, the boolean (jaccard/dice/yule/…) family, hellinger,
and custom callables stay CPU-fallback. Under device=GPU the standard-branch
kNN graph is **exact** up to n = 100k and **GPU NN-descent** above — quality
parity like the rest of GPU mode, deliberately *not* bit-identical to
Python's approximate graph (and, uniquely on the GPU-NN-descent leg,
not bit-identical between two runs either: reverse-candidate sampling
races by design, the same family as the hogwild SGD). GPU standard fits
build no search index: transform runs its kNN on the engine directly at any
n (exact, O(n_q·n)), `save()`/`load()` round-trip (the index sections are
optional), but `update()` raises and a saved model needs Metal at load-site
to transform.

kNN timings (M1 Max 32-core, k 15; hidden tests `-ns -tc="gpu knn
crossover timing*"`, remeasured 2026-08-30 after the P1 kernel rewrite — the
register-blocked decomposable-metric fast body, ~16–18× over the old scalar
engine):

| n | GPU exact (dim 16) | GPU nnd (dim 16) | GPU exact (dim 64) | GPU nnd (dim 64) | nnd recall (16 / 64) |
| --- | --- | --- | --- | --- | --- |
| 20k | 35 ms | 143 ms | 74 ms | 507 ms | 0.997 / 0.973 |
| 30k | 51 ms | 214 ms | 187 ms | 818 ms | 0.997 / 0.959 |
| 50k | 147 ms | ~385 ms | ~720 ms | ~1530 ms | 0.995 / 0.934 |
| 100k | ~520 ms | ~990 ms | ~2650 ms | ~3880 ms | 0.993 / 0.893 |

The rewritten exact engine is now faster than GPU NN-descent at **every**
measured size and both dims through 100k — while still returning recall-1.0
graphs — so the auto crossover (`umap::metal::kGpuNndCrossoverN`) moves from
30000 to **100000**: exact owns the whole measured range, NN-descent takes
over only in the truly-large, unmeasured regime where the exact engine's O(n²)
term must eventually overtake the descent's ~O(n log n) (it has not by 100k on
either dim) and beyond a single Metal buffer. The descent's recall still
tracks the bit-exact CPU NN-descent port's to ~1e-4 at each size (it falls off
with dim/n as shown, which the exact engine sidesteps entirely below the
crossover). `UMAP_GPU_KNN_MODE=exact|nnd` overrides the mode for experiments.
(The
2026-08-29 gpu-nnd pass also fixed a silent-corruption hazard in the exact
engine: monolithic multi-second dispatches at n ≈ 100k were sporadically
aborted by the macOS GPU watchdog and returned zero-filled buffers unnoticed
— queries now run in work-budgeted slices and every command buffer's status
is checked, falling back to the CPU on failure. The layout SGD engine
carries the same hardening: epoch batches are sized adaptively from the
measured per-epoch time (~250 ms per submission, 25-epoch cap) and a failed
buffer aborts with the host embedding untouched, so the stderr-noted CPU
fallback reruns from the initial embedding, never a partially-optimized
one.)

GPU results are validated at embedding-quality level (per-metric kNN parity
vs a float64 CPU reference at ≤ 1e-5, NN-descent recall ≥ 0.90 asserted at
median-of-3 vs the exact engine, trustworthiness parity with CPU fits
and transforms — incl. a 40k GPU-NN-descent fit within 0.05 of a CPU
NN-descent fit — held-out transform accuracy, densMAP density-order parity —
rad_emb Spearman ≥ 0.9 vs the seeded CPU run, inside the CPU reference's own
sequential-vs-hogwild band), **not** bitwise: kNN distances are float32
(exact ties resolve in ascending data-index order), Metal has no float64,
the device RNG is seeded per (edge, epoch), and hogwild/sampling updates
race by design. The complete remaining CPU-only surface under device=GPU:
RandomState-stream-critical internals (`update()`, supervised target-y kNN,
`inverse_transform` ranking, `output_dens` epilogue kNN, composites),
n_epochs lists (embedding snapshots), non-euclidean output metrics,
exotic/discrete/boolean/callable input metrics, and k > 256. Users who need
seeded bit-exact parity with the Python reference should keep the default
`Device::CPU`.

### Penalized kNN (auxiliary GPU kernel)

[`include/umap/metal_knn.h`](include/umap/metal_knn.h) exposes an extra
utility kernel that ships inside the same `umap_kernels.metallib` but is
**not used by the UMAP pipeline itself**: an exact brute-force kNN whose
distance is a cosine or euclidean base over F ≤ 8 feature dims plus weighted
penalties `wt·|dz_i−dz_j| + wp·|cz_i−cz_j| + wr·|rz_i−rz_j| +
wk·[tid_i≠tid_j]` (z-scored day/close/return columns and a ticker-mismatch
constant, e.g. for cross-sectional financial panels). The API is global-scope
C++ (no namespace):

```cpp
#include <umap/metal_knn.h>

bool gpu_knn_available();
bool gpu_knn(const float* X, const float* sq, const float* dz, const float* cz,
             const float* rz, const int32_t* tid, int n, int F, int k,
             bool cosine, float wt, float wp, float wr, float wk,
             int32_t* outI, float* outD);
```

`X` is n×F row-major (rows unit-normalized when `cosine=true`), `sq` the
squared row norms (euclidean base). Outputs are n×k rows sorted ascending by
penalized distance, self excluded, padded with index −1 / distance 1e30 past
the n−1 real neighbours. Requires F ≤ 8 and k ≤ 64; anything else — or no
Metal device, or an `UMAP_ENABLE_METAL=OFF` build (a CPU stub keeps callers
linking) — returns `false` and the caller runs its own CPU fallback.

## Tests

`./build/umap_tests` — doctest port of the entire upstream test suite
(`umap/tests/*.py`, thresholds and seeds verbatim) plus golden-value parity
suites generated from the real Python implementation
(`scripts/gen_golden*.py`, `scripts/gen_upstream_fixtures.py`):

* Metal build: **432 test cases / 520,013 assertions**, 0 failures
  (26 skips: version-removed scipy metrics, upstream `@skip` mirrors,
  typed-surface-unrepresentable cases, four hidden GPU timing cases);
* CPU-only build: **407 cases / 174,687 assertions**, 0 failures (22 skips).

`scripts/test_install.sh` additionally verifies `cmake --install`, the
`find_package(umapcpp)` consumer build (C++ and C), and the installed CLI
(CSV + npy + model round trip) end to end.

## Out of scope

Intentionally not ported (Python-ecosystem-bound; documented here and in
[ROADMAP.md](ROADMAP.md)):

* **ParametricUMAP** — a TensorFlow/Keras neural-network embedder; porting it
  would mean reimplementing a deep-learning stack, and its behaviour is
  defined by TF internals rather than the UMAP algorithm.
* **`umap.plot`** and the Bokeh/interactive tooling — matplotlib/pandas/
  datashader front-ends; use the CSV/npy outputs with your plotting tool of
  choice.

## Layout

```
include/umap/       public headers (umap.hpp is the main entry; umap_c.h C API)
src/                library implementation (+ src/metal/ GPU kernels)
apps/umap_cli.cpp   the `umap` executable
tests/              doctest suites + golden headers
scripts/            golden/fixture generators (need the reference venv),
                    test_install.sh
examples/consumer/  find_package(umapcpp) smoke project
reference/          shallow clone of lmcinnes/umap (gitignored)
third_party/        vendored metal-cpp, qhull_r (scipy-exact), doctest
ROADMAP.md          feature-by-feature status, parity levels, deviations log
```

## License

The port follows the licensing of the code it reproduces: UMAP (BSD
3-Clause, © Leland McInnes) and pynndescent (BSD 2-Clause); vendored qhull
sources carry the qhull license, metal-cpp Apache 2.0, doctest MIT.
