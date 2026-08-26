# UMAP C++ / Metal Port — Roadmap & Progress

Goal: a complete, faithful C++ reproduction of UMAP (lmcinnes/umap), GPU-parallelizable
via Apple Metal, passing equivalent tests, consumable as a static library + unix CLI executable.

Reference source: `reference/umap/` (shallow clone of lmcinnes/umap, gitignored).

## Status legend
- [ ] not started  [~] in progress  [x] done (implemented + tested)

## Phase 1 — Core CPU library (faithful port)
- [~] CMake scaffold: `libumapcpp` static lib, `umap` CLI, `umap_tests` (doctest, vendored header) — lib + tests done; CLI deferred to Phase 4
- [x] Distance metrics (euclidean, sqeuclidean, manhattan, chebyshev, minkowski, cosine, correlation, hamming, jaccard, dice, hellinger, canberra, braycurtis + named_distances registry) — port of `umap/distances.py`; `_grad` variants deferred until output-metric layouts need them
- [x] RNG: port of tau_prng (xorshift used in `umap/utils.py`), bit-exact incl. negative seeds
- [x] RP-tree forest init + NN-descent kNN (port of pynndescent core path used by UMAP) — `include/umap/nn_descent.hpp` + `src/nn_descent.cpp`: heaps, new_build_candidates, euclidean/angular RP-tree splits + forest leaf arrays, init_rp_tree/init_random, array-update NN-descent loop, `nearest_neighbors` toplevel with UMAP's hacked n_trees/n_iters; bit-exact numpy RandomState seeding; index-identical to Python end-to-end (single-threaded)
- [x] `smooth_knn_dist` (sigma/rho binary search, SMOOTH_K_TOLERANCE, MIN_K_DIST_SCALE, bandwidth, local_connectivity interpolation, mean-distance fallbacks)
- [x] `compute_membership_strengths` + `fuzzy_simplicial_set` (incl. set_op_mix_ratio union/intersection mix, -1 knn masking, return_dists; CSR sparse matrix in `include/umap/sparse_matrix.hpp`)
- [x] `find_ab_params` (Levenberg-Marquardt fit of 1/(1+a x^{2b}))
- [ ] Spectral initialization (normalized Laplacian eigenvectors; Eigen + Spectra), random init fallback
- [x] `simplicial_set_embedding` / `optimize_layout_euclidean` SGD (negative sampling, epochs_per_sample) — clip/rdist/make_epochs_per_sample + full euclidean SGD (per-head-vertex rng states, move_other, alpha decay) in `include/umap/layouts.hpp` + `src/layouts.cpp`, bit-exact vs the sequential numba path; non-init glue as `simplicial_set_embedding_from_init` (n_epochs default 500/200 @ n>10000, weight filtering + eliminate_zeros, [0,10] rescale, move_other=true); spectral init remains an input seam; densmap path TODO; optional std::thread hogwild `parallel` mode
- [ ] `UMAP` class: fit / fit_transform / transform (transform = kNN of new points vs training, init by weighted mean, constrained SGD)
- [ ] Sparse input support (CSR)

## Phase 2 — Tests (port of umap/tests)
- [ ] Test harness + small datasets (iris baked in; synthetic clusters; nn data)
- [ ] test_umap_nn equivalents (kNN accuracy ≥ thresholds)
- [ ] test_umap_metrics equivalents (metric correctness vs scipy-computed golden values)
- [ ] test_umap_trustworthiness equivalents (trustworthiness ≥ thresholds on iris etc.)
- [ ] test_umap_on_iris equivalents
- [ ] test_umap_ops / repeated-data / validation-params equivalents
- [ ] Golden-value comparison vs Python UMAP outputs (scripts/gen_golden.py)

## Phase 3 — Apple Metal GPU parallelization
- [x] Metal compute kernels: pairwise/kNN distance computation — `brute_force_knn` (euclidean + cosine, per-thread top-k ≤ 64, exact vs CPU kNN) in `src/metal/umap_kernels.metal`, host API `include/umap/metal/gpu_knn.hpp`; metal-cpp vendored in `third_party/metal-cpp`, shaders compiled to `umap_kernels.metallib` at build time (xcrun metal/metallib custom commands), all gated behind `UMAP_ENABLE_METAL` (default ON on APPLE)
- [x] Metal SGD layout optimization (edge-parallel hogwild matching numba prange race semantics) — `sgd_epoch` kernel: one thread per 1-simplex, unsynchronized f32 embedding updates, device-buffer epoch bookkeeping (f32; Metal has no f64 — documented deviation), device tau_rand_int (exact rng.hpp 64-bit math) seeded per (edge, epoch) via splitmix64, host-side alpha decay, ±4 clip; epochs batched 25/command buffer, edges split into 32 sub-dispatches per epoch to bound concurrent racy updates per vertex (~degree/32 — full-grid dispatch measurably degraded small-graph quality); API `include/umap/metal/gpu_layout.hpp`
- [x] CPU/GPU parity tests (quality-level, not bitwise) — `tests/test_metal.cpp` (skips gracefully without a Metal device): GPU kNN indices identical (ties aside) and dists ≤ 1e-5 vs CPU exact on 500×8 (+ distinct-query case); GPU SGD on same graph+init: finite/bounded embeddings, two-gaussian centroid-separation criterion, 3-cluster intra≪inter proxy for both CPU and GPU (GPU inter/intra ~5–7 vs CPU-sequential ~11, CPU-hogwild ~7–10); hidden 5k-point timing case (`-ns -tc="gpu sgd timing*"`): GPU ~9× vs sequential CPU (191 ms vs 1724 ms, 88k edges × 200 epochs)
- [~] Runtime dispatch: `--device cpu|gpu` — `include/umap/device.hpp` seam (Device enum, metal_available(), resolve_device) landed; wiring into the UMAP class/CLI once those exist (Phase 1/4)

## Phase 4 — Packaging
- [ ] CLI: `umap fit --input data.csv|.npy --output emb.csv ...` with all main hyperparams
- [ ] Install target + umapcpp-config.cmake so other projects can `find_package(umapcpp)`
- [ ] C API header (umap_c.h) for FFI consumers
- [ ] README with usage

## Decisions
- C++17, Apple Silicon primary target; Metal via metal-cpp.
- Eigen (FetchContent) for linear algebra; Spectra for eigensolver; doctest for tests.
- Faithfulness bar: same algorithms/constants/defaults as Python; stochastic outputs
  validated via trustworthiness/kNN-accuracy thresholds like the upstream test suite,
  plus deterministic sub-component golden tests (smooth_knn, membership strengths, metrics, a/b fit).

## Log
- 2026-08-24: repo initialized, reference cloned, roadmap written. Phase 1 kickoff.
- 2026-08-24: Phase 1 foundations landed: CMake build (static lib + doctest tests), 13 distance metrics + registry, tau_prng (bit-exact vs numba), CSR sparse matrix, smooth_knn_dist / compute_membership_strengths / fuzzy_simplicial_set (float32-faithful), find_ab_params (LM). Goldens generated from the real reference impl via scripts/gen_golden.py (venv with numba/pynndescent). 34 test cases / 998 assertions, all green.
- 2026-08-25: RP-tree forest + NN-descent kNN landed (dense low_memory path UMAP's fit uses): utils.py heaps (simple/checked/checked_flagged push, siftdown, deheap_sort), new_build_candidates with virtual-thread rng semantics, euclidean+angular random projection splits, make_forest leaf arrays, init_rp_tree (n_threads=8 block partitioning) + init_random, array-update nn_descent loop (n_threads=1 reproducible mode), nearest_neighbors toplevel (n_trees=min(64,5+round(sqrt(N)/20)), n_iters=max(5,round(log2 N)), max_candidates=60), fast-alternative metrics (sqeuclidean+sqrt correction, alternative_cosine+2^-d correction) with metric-driven angular tree selection, bit-exact NumpyRandomState (MT19937 masked rejection) for np.random.RandomState seeding parity. Goldens via scripts/gen_golden_nn.py (heap ops, candidate building, splits, whole-tree leaves, RandomState draws). Verified vs Python: kNN indices 100% identical end-to-end on 80x4 and 600x8 (euclidean + cosine, seed 42, NUMBA_NUM_THREADS=1), dists equal to ~1 ulp (fastmath); recall vs brute force 1.0 / 0.998 / 0.999. prepare()/search-graph/diversify not ported (UMAP fit only consumes _neighbor_graph); transform() query path TODO. 56 test cases / 17364 assertions, all green.
- 2026-08-25 (branch metal-gpu): Apple Metal GPU kernels landed: vendored metal-cpp (Apple macOS14.2 release) in third_party/, .metal → .metallib build pipeline (xcrun custom commands, UMAP_DEFAULT_METALLIB compile def + UMAP_METALLIB_PATH env override), UMAP_ENABLE_METAL gate (CPU-only config verified green). Kernel 1: exact brute-force kNN (euclidean+cosine, insertion top-k ≤ 64) — identical indices / ≤1e-5 dists vs CPU on 500×8. Kernel 2: optimize_layout_euclidean epoch loop — 1 thread/edge hogwild (numba-prange race semantics), f32 bookkeeping in device buffers (no f64 on Metal), device-side bit-exact tau_rand_int seeded per (edge, epoch) via splitmix64 (distinct odd strides — gamma-collision seeding bug found+fixed), host alpha decay, 32 edge sub-dispatches/epoch to bound concurrent racy updates (full-grid simultaneity degraded small-graph quality: inter/intra 2.1 → 5–7, CPU-hogwild 7–10). GPU ~9× vs sequential CPU on 5k pts (191 ms vs 1724 ms, 88k edges, 200 epochs). device.hpp dispatch seam for the future UMAP class. 60 test cases (4 new GPU) all green with Metal; 56 green without.
- 2026-08-24 (branch layout-sgd): layout SGD landed: optimize_layout_euclidean bit-exact vs the sequential numba reference over 50- and 500-epoch goldens (per-head-vertex rng_state_per_sample seeding incl. int64 bitcast of embedding[:,0], Python-modulo negative sampling, exact f4/f8 promotion order), make_epochs_per_sample bit-exact (float32 n_samples, float64 result), simplicial_set_embedding non-init glue (threshold filter, [0,10] float32 rescale, move_other=true), std::thread hogwild parallel mode mirroring numba prange, qualitative two-cluster separation tests. Goldens via scripts/gen_golden_layout.py. 42 test cases / 1955 assertions, all green. densmap objective and spectral init still TODO (init is a seam of simplicial_set_embedding_from_init).
