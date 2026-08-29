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
(trustworthiness / kNN accuracy / ARI, ported verbatim).

Documented deviations (each noted at its definition site and in
[ROADMAP.md](ROADMAP.md)): Spectra-vs-ARPACK eigensolver residuals in spectral
init (~1e-6, up to sign), `find_ab_params` reaching scipy's optimum to ~1e-7,
tau-prng standing in for `RandomState` in some multi-component spectral
branches, exact-tie ordering in a few argsort/argpartition sites, and the GPU
path's float32/hogwild semantics (below). `ROADMAP.md` is the authoritative,
per-feature log of parity levels and deviations.

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

## GPU (Apple Metal)

`UMAPParams::device = umap::Device::GPU` (CLI: `--device gpu`) routes to Metal
kernels where they genuinely cover the computation:

* the layout SGD of standard euclidean-output fits (one thread per edge,
  hogwild float32 updates — the same race semantics as numba's
  `parallel=True`; ~9× vs the sequential CPU loop on a 5k-point benchmark);
* exact brute-force kNN where the small-data branch (and transform) would
  compute sklearn-replica pairwise distances, for dense euclidean/cosine with
  `n_neighbors ≤ 64`.

Everything else (densMAP, non-euclidean output metrics, n_epochs lists,
transform's frozen-embedding SGD, sparse/precomputed kNN) falls back to the
CPU port with a stderr note. GPU results are validated at embedding-quality
level (trustworthiness parity with CPU), **not** bitwise: Metal has no
float64, the device RNG is seeded per (edge, epoch), and hogwild updates race
by design. Users who need seeded bit-exact parity with the Python reference
should keep the default `Device::CPU`.

## Tests

`./build/umap_tests` — doctest port of the entire upstream test suite
(`umap/tests/*.py`, thresholds and seeds verbatim) plus golden-value parity
suites generated from the real Python implementation
(`scripts/gen_golden*.py`, `scripts/gen_upstream_fixtures.py`):

* Metal build: **368 test cases / 172,124 assertions**, 0 failures
  (27 skips: version-removed scipy metrics, upstream `@skip` mirrors,
  typed-surface-unrepresentable cases, a hidden GPU timing case);
* CPU-only build: **362 cases / 169,712 assertions**, 0 failures (26 skips).

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
* AlignedUMAP is tracked in ROADMAP.md (in progress on a separate branch).

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
