# Upstream test-suite port — notes (Phase 2, branch tests-port)

Port of `reference/umap/umap/tests/*.py` to doctest. One `TEST_SUITE` per
upstream file; upstream test names are preserved as `TEST_CASE` names.
Skipped upstream tests exist as `doctest::skip()`-marked stubs so the suite
structure mirrors pytest's (and skips are visible in the test counts).

Fixtures: `scripts/gen_upstream_fixtures.py` reproduces the conftest.py
session fixtures exactly (`np.random.seed(189212)`, draws in conftest
definition order: spatial_data, binary_data, nn_data, binary_nn_data,
sparse_nn_data, iris_selection) into `tests/upstream_data.hpp`; iris is the
real `sklearn.datasets.load_iris` data. Metric goldens
(`tests/upstream_metric_goldens.hpp`) are sklearn/scipy/BallTree matrices with
the upstream tests' non-finite corrections pre-applied. Grad-test sampler
draws (`np.random.default_rng(0)`) are in `tests/upstream_grad_data.hpp`.
Venv used: scipy 1.18.1, sklearn 1.9.0, numpy 2.5.2, numba 0.67.0.

## Whole files skipped

| upstream file | reason |
|---|---|
| test_parametric_umap.py | TensorFlow/Keras — out of C++ scope |
| test_plot.py | matplotlib — out of C++ scope |
| test_aligned_umap.py | AlignedUMAP not yet ported |
| test_composite_models.py | fuzzy-set union/intersection composite models not ported |
| test_spectral.py | both tests target `tswspectral_layout` (init="tswspectral" is TODO); the ported `spectral_layout` surface is already covered by tests/test_spectral.cpp (eigenpair parity, multi-component layouts, dispatch) |

## Unskip pass (branch tests-unskip)

After densmap/output_dens, supervised fit(y), the sparse UMAP-class wiring,
output_metric/metric_kwds and the full sparse metric registry landed, every
skip-stub those features unblocked was flipped into a real ported test
(32 stubs) and the previously whole-file-skipped test_densmap.py was ported
(3 active cases, as a TEST_SUITE inside tests/test_densmap.cpp): the sparse
+ 5 portable supervised trustworthiness cases, the sphere/sparse/supervised
iris legs, the 4 sparse repeated-data variants, the densmap/output_metric/
target_n_neighbors validation cases (7), and the 12 remaining sparse metric
cases; test_umap_fit_instance_returned also gained its fit(x, y) leg. All
upstream thresholds verbatim;
goldens for the new sparse metric cases were added to
scripts/gen_upstream_fixtures.py (purely additive regeneration -- existing
arrays byte-identical). The remaining skips and their reasons are documented
per file below.

### test_densmap.cpp (TEST_SUITE "test_densmap.py")
- `test_densmap_trustworthiness_random_init` is `@pytest.mark.skip` upstream;
  mirrored.
- `test_densmap_trustworthiness_on_iris`: the transform NotImplementedError
  leg maps to the C++ std::runtime_error with the upstream message; the
  inverse_transform(ValueError) leg is omitted (inverse_transform is in
  progress on main).
- fit_transform's upstream (embedding, rad_h, rad_l) tuple maps to
  fit_transform + rad_orig()/rad_emb() accessors.

## Per-file skips / adaptations

### test_umap_trustworthiness.cpp
- Skip: `test_string_metric_supervised_umap_trustworthiness` -- upstream's y
  is an object array of strings, unrepresentable against the float-typed C++
  y surface (target_metric="string" raises here, matching the port's
  documented treatment of object-dtype targets).
- Ported (tests-unskip): `test_umap_sparse_trustworthiness` (conftest's
  sparse_test_data = csr(nn_data * binary_nn_data); the first 100 rows'
  product built densely and CSR-converted -- identical matrix; unseeded fit
  like upstream) and the supervised/semi-supervised/metric(l1)/ordinal/count
  cases on blobs50 + blobs50_labels. Thresholds verbatim.
- `test_sparse_precomputed_metric_umap_trustworthiness`: the precomputed dmat
  is passed dense (sparse precomputed not ported); the matrix upstream wraps
  in CSR is fully dense anyway. Thresholds verbatim.

### test_umap_on_iris.cpp
- Skips: modified_dtype (no dtype to modify in C++), sparse precomputed
  transform (the sparse precomputed branch is not ported).
- Ported (tests-unskip): sphere iris (output_metric="haversine"; the 3D
  sphere projection + cosine-metric trustworthiness follow the upstream
  formulas -- trustworthiness.hpp gained the input-space "cosine" option),
  sparse transform (CSR fit/transform on the iris subsets), supervised
  clusterability (KMeans(3) + adjusted_rand_score via the
  upstream_helpers.hpp ports; upstream leaves KMeans unseeded).
- `test_umap_inverse_transform_on_iris`: active; sklearn KDTree queries
  ported as exact brute-force kNN.
- `test_precomputed_knn_on_iris`: pytest.warns bookkeeping replaced by
  behavioural checks (identical embeddings, transform raising with
  "search index" in the message); the C++ port prints the same warning texts
  to stderr. Thresholds (rms < 0.005) verbatim.

### test_umap_nn.cpp
- The accuracy tests are `@pytest.mark.skip` upstream but run ACTIVELY here
  with upstream's thresholds (0.85/0.89/0.85/0.75/0.85/0.90), seed 42 standing
  in for upstream's `np.random` random_state. low_memory=True is the only
  ported NN-descent path, so the low_memory variants exercise the same code.
- Skip: callable-metric accuracy test.

### test_umap_metrics.cpp
- Version skips mirrored from the venv (scipy 1.18 / sklearn 1.9):
  kulsinski (+sparse), sokalmichener (+sparse), sparse_sokalsneath.
- Ported (tests-unskip): the full sparse metric surface -- sparse chebyshev /
  minkowski (numba default p=2.0, matching upstream's kernel call without p) /
  hamming / canberra / braycurtis / correlation, the sparse binary metrics
  (jaccard, matching, dice, rogerstanimoto, russellrao; csr(binary_data) with
  1.0f values), and sparse_hellinger (rows [:-2], |values|, decimal=3).
  sparse_need_n_features metrics get n_features=20 bound like pynndescent's
  _partial_dist_func. The sparse_ll_dirichlet leg evaluates the full 12x12
  matrix on the raw signed data like upstream, which only asserts the result
  "is not None" -- NaN entries (log of negative sums) are permitted there, so
  no per-entry value check.
- Tolerances are upstream's `assert_array_almost_equal` decimals (6; 5 for the
  mahalanobis grad golden; 3 for sparse hellinger).

### test_umap_ops.cpp
- Skips: component_layout_options (precomputed-metric + linkage
  component_layout not ported), the sparse=True leg of
  disconnected_data_precomputed. update()/update_large are active (the
  upstream vstack/hstack graph rearrangement is computed densely).
- `test_disconnected_data`: warning-count assertions replaced by the model's
  `vertices_disconnected()` bookkeeping (same event, warnings go to stderr).
- `test_fast_knn_indices_matches_argsort` draws its (distinct-valued) random
  matrix from std::mt19937 — the property is RNG-agnostic. The ties matrix of
  the second test IS upstream's RandomState(1) matrix (baked `tie_D`).
- Upstream-unseeded datasets (make_blobs in blobs_cluster / multi_component /
  graph_layout, np.random.choice disconnection data, transform noise) are
  baked with documented fixed seeds (random_state=42 / RandomState(42) /
  RandomState(7)); the UMAP fits themselves stay unseeded like upstream.

### test_umap_repeated_data.cpp
- All eight cases run (tests-unskip flipped the four sparse variants):
  conftest's sparse_spatial_data_repeats = csr(spatial_repeats *
  binary_repeats) and sparse_binary_data_repeats = csr(binary_repeats) are
  built from the baked fixtures. All cases verbatim (n_neighbors=3,
  unique=True, upstream n_epochs values, unseeded).

### test_data_input.cpp
- Upstream's numba `nan_dist` callable is not portable; the euclidean metric
  with init="random", n_epochs=11 is used instead — the `ensure_all_finite`
  check_array gate under test fires before any distance work in both.
  update()/inverse_transform() legs skipped (not ported).

### test_umap_validation_params.cpp
- Ported: every case expressible against the typed C++ params surface
  (set_op_mix_ratio, min_dist, n_components<=0, n_neighbors<2, unknown
  metric, learning_rate, repulsion_strength, negative_sample_rate, bad
  init string, ndarray-init shape, n_epochs<0, haversine on high-d data,
  a/b passthrough + n_neighbors truncation, unique+precomputed,
  hellinger negative data, fit-returns-self). tests-unskip added
  target_n_neighbors<2, the three bad output_metric legs, haversine
  embed-to-3d (raised at output-metric resolution like the upstream fit),
  the densmap dens_lambda/dens_var_shift/dens_frac knobs and
  densmap+haversine, and the fit(x, y) leg of fit_instance_returned.
- Skips: values unrepresentable in the typed surface (float/str n_components,
  numeric metric/init objects, y length mismatches against the pointer-typed
  y), n_jobs, callable metrics. test_umap_update_bad_params and
  test_umap_inverse_transform_fails_expectedly are active; densmap knobs and
  output_metric validation legs are active (tests-unskip).
- `test_umap_fit_instance_returned` uses n_epochs=11/init="random" for
  runtime; the assertion (fit returns the instance) is unaffected.

### test_umap_get_feature_names_out.cpp
- The C++ `get_feature_names_out()` takes no `input_features` argument (it is
  ignored upstream), so the first case reduces to the default case.
- Skip: FeatureUnion/Pipeline case (sklearn machinery).

### test_chunked_parallel_spatial_metric.cpp
- Only the hellinger implementation-consistency test is live upstream (all
  else is `@benchmark_only`; mirrored as skips). The C++ port has no
  chunked/parallel evaluator, so the check becomes: symmetric-fill vs
  rectangular vs hellinger_grad float32 matrices identical, and equal to the
  reference `pairwise_special_metric` goldens. The unseeded
  `spatial_data_two` operand is replaced by a deterministic row permutation.

### test_umap_grads.cpp
- Full port of `assert_gradient_matches_finite_diff` (eps, tolerances,
  forward_only and skip_close_coords flags verbatim); pytest dim/p
  parametrizations run as in-case loops. `test_mahalanobis_gradient` diffs a
  local float64 mahalanobis exactly like upstream's `dist.mahalanobis_f64`.

## Library changes on this branch (audited against upstream)

- `src/nn_descent.cpp` / `include/umap/nn_descent.hpp`: dense NN-descent
  support for `jaccard` and `hellinger` — line-faithful ports of
  pynndescent's `alternative_jaccard` / `correct_alternative_jaccard` /
  `alternative_hellinger` / `correct_alternative_hellinger` (float32 locals)
  plus the plain kernels; both metrics are in pynndescent's angular-trees
  list. Required by `test_disconnected_data` with
  force_approximation_algorithm=True.
- `src/umap.cpp` / `include/umap/umap.hpp`: `ensure_all_finite`
  (sklearn check_array semantics, upstream fit/transform kwarg; exposed as a
  param), the umap_.py:1932 "Metric 'hellinger' does not support negative
  values" fit guard, and `get_feature_names_out()`
  (ClassNamePrefixFeaturesOutMixin naming). Required by test_data_input.py,
  test_umap_validation_params.py and test_umap_get_feature_names_out.py.
- `tests/test_umap.cpp`: its local trustworthiness helper factored into
  `tests/trustworthiness.hpp` (shared with the new suites); no behaviour
  change.
