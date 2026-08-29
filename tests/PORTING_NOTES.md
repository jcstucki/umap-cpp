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
| test_densmap.py | densmap in progress on main (separate workstream) |
| test_composite_models.py | fuzzy-set union/intersection composite models not ported |
| ~~test_spectral.py~~ | PORTED (branch tswspectral) as tests/test_tswspectral.cpp: both upstream cases verbatim — `test_tsw_spectral_init` (rmsd < 1e-6, per-column sign aligned; the upstream test is skipped for scipy >= 1.15 because eigsh "SM" @ tol=1e-4 misconverges there, and the C++ solvers behave like the scipy [1.10, 1.15) window where it runs) and `test_ensure_fallback_to_random_on_spectral_failure` (unseeded upstream `u` baked as np.random.seed(777), regenerated bit-exact in-test via NumpyRandomState; pytest.warns → `used_random_fallback` out-param). The `spectral_layout` surface is covered by tests/test_spectral.cpp |

## Per-file skips / adaptations

### test_umap_trustworthiness.cpp
- Skips: `test_umap_sparse_trustworthiness` (sparse input not wired into the
  UMAP class), the six supervised/semi-supervised cases (fit(y) not ported).
- `test_sparse_precomputed_metric_umap_trustworthiness`: the precomputed dmat
  is passed dense (sparse precomputed not ported); the matrix upstream wraps
  in CSR is fully dense anyway. Thresholds verbatim.

### test_umap_on_iris.cpp
- Skips: sphere iris (output_metric not wired into the class), modified_dtype
  (no dtype to modify in C++), sparse transform + sparse precomputed
  transform (sparse class glue not ported), supervised clusterability,
  inverse_transform.
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
- Skips for unported sparse metrics: sparse chebyshev / minkowski / hamming /
  canberra / braycurtis / correlation, all sparse binary metrics,
  sparse_hellinger (the C++ sparse registry ports euclidean/manhattan/cosine).
- Tolerances are upstream's `assert_array_almost_equal` decimals (6; 5 for the
  mahalanobis grad golden; 3 for sparse hellinger — n/a).

### test_umap_ops.cpp
- Skips: update()/update_large (update not ported), component_layout_options
  (precomputed-metric + linkage component_layout not ported), the sparse=True
  leg of disconnected_data_precomputed.
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
- Skips: the four sparse variants (sparse class glue not ported). Dense cases
  verbatim (n_neighbors=3, unique=True, upstream n_epochs values).

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
  hellinger negative data, fit-returns-self).
- Skips: values unrepresentable in the typed surface (float/str n_components,
  numeric metric/init objects), unported features (output_metric,
  target_n_neighbors, densmap knobs, n_jobs, callable metrics, update,
  inverse_transform, fit(X, y) length checks).
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
