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
- [ ] Metal compute kernels: pairwise/kNN distance computation
- [ ] Metal SGD layout optimization (epoch-parallel with atomic/lock-free updates, matching numba parallel semantics)
- [ ] CPU/GPU parity tests (trustworthiness parity, not bitwise)
- [ ] Runtime dispatch: `--device cpu|gpu`

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
- 2026-08-24 (branch layout-sgd): layout SGD landed: optimize_layout_euclidean bit-exact vs the sequential numba reference over 50- and 500-epoch goldens (per-head-vertex rng_state_per_sample seeding incl. int64 bitcast of embedding[:,0], Python-modulo negative sampling, exact f4/f8 promotion order), make_epochs_per_sample bit-exact (float32 n_samples, float64 result), simplicial_set_embedding non-init glue (threshold filter, [0,10] float32 rescale, move_other=true), std::thread hogwild parallel mode mirroring numba prange, qualitative two-cluster separation tests. Goldens via scripts/gen_golden_layout.py. 42 test cases / 1955 assertions, all green. densmap objective and spectral init still TODO (init is a seam of simplicial_set_embedding_from_init).
