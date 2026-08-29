cmake_minimum_required(VERSION 3.16)
project(umapcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(umapcpp STATIC
    src/distances.cpp
    src/sparse_matrix.cpp
    src/fuzzy_simplicial_set.cpp
    src/find_ab_params.cpp
)
# phase4: BUILD_INTERFACE/INSTALL_INTERFACE split so the exported target
# (find_package(umapcpp)) resolves headers from the install prefix.
target_include_directories(umapcpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(umapcpp PRIVATE -O2 -Wall -Wextra -Wpedantic)

# --- layout-sgd additions (branch layout-sgd) -------------------------------
target_sources(umapcpp PRIVATE src/layouts.cpp)
find_package(Threads REQUIRED)  # hogwild parallel mode (std::thread)
target_link_libraries(umapcpp PUBLIC Threads::Threads)
# ----------------------------------------------------------------------------

# --- metrics-full additions -------------------------------------------------
target_sources(umapcpp PRIVATE src/layouts_generic.cpp)
# ----------------------------------------------------------------------------

# --- nn-descent additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/nn_descent.cpp)
# ----------------------------------------------------------------------------

# --- query-path additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/nn_query.cpp)
# ----------------------------------------------------------------------------

# --- sparse-input additions -------------------------------------------------
target_sources(umapcpp PRIVATE src/sparse_distances.cpp src/sparse_nn_descent.cpp)
# ----------------------------------------------------------------------------

# --- sparse-transform additions ---------------------------------------------
target_sources(umapcpp PRIVATE src/sparse_nn_query.cpp)
# ----------------------------------------------------------------------------

# --- spectral-init additions ------------------------------------------------
# Eigen + Spectra, header-only via FetchContent. SOURCE_SUBDIR points at a
# directory without a CMakeLists.txt so MakeAvailable only populates (no
# add_subdirectory of their build systems).
include(FetchContent)
FetchContent_Declare(eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_SUBDIR cmake_disabled_do_not_add_subdir
)
FetchContent_Declare(spectra
    URL https://github.com/yixuan/spectra/archive/refs/tags/v1.0.1.tar.gz
    SOURCE_SUBDIR cmake_disabled_do_not_add_subdir
)
FetchContent_MakeAvailable(eigen3 spectra)
target_sources(umapcpp PRIVATE src/spectral.cpp)
# phase4: BUILD_INTERFACE-wrapped so the exported target carries no build-tree
# paths (no public umap header includes Eigen/Spectra; they are build-only).
target_include_directories(umapcpp SYSTEM PUBLIC
    $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}>
    $<BUILD_INTERFACE:${spectra_SOURCE_DIR}/include>
)
# ----------------------------------------------------------------------------

# --- umap-class additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/umap.cpp)
# ----------------------------------------------------------------------------

# --- phase4 additions --------------------------------------------------------
# Model persistence (UMAP::save/load + PreparedIndex::serialize/deserialize)
# and the C API (include/umap/umap_c.h).
target_sources(umapcpp PRIVATE src/model_io.cpp src/umap_c.cpp)
# ------------------------------------------------------------------------------

# --- supervised-sparse additions ---------------------------------------------
target_sources(umapcpp PRIVATE src/supervised.cpp)
# ----------------------------------------------------------------------------

# --- aligned-umap additions --------------------------------------------------
target_sources(umapcpp PRIVATE src/aligned_umap.cpp src/layouts_aligned.cpp)
# ----------------------------------------------------------------------------

# --- densmap additions ------------------------------------------------------
# densMAP replicates np.dot(float32, float32) via the same BLAS numpy's macOS
# wheels link (Accelerate); off Apple platforms a sequential f32 dot is used.
if(APPLE)
    target_link_libraries(umapcpp PUBLIC "-framework Accelerate")
endif()
# ----------------------------------------------------------------------------

# --- metal-gpu additions ----------------------------------------------------
# Apple Metal GPU kernels (brute-force kNN + SGD layout). Gated so CPU-only
# builds (and non-Apple platforms) keep working: -DUMAP_ENABLE_METAL=OFF.
if(APPLE)
    option(UMAP_ENABLE_METAL "Build the Apple Metal GPU kernels" ON)
else()
    option(UMAP_ENABLE_METAL "Build the Apple Metal GPU kernels" OFF)
endif()

target_sources(umapcpp PRIVATE src/device.cpp)  # CPU fallback of metal_available()

if(UMAP_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "UMAP_ENABLE_METAL requires an Apple platform")
    endif()

    # Compile the .metal shaders into a .metallib at build time.
    set(UMAP_METAL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/umap_kernels.metal)
    set(UMAP_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/umap_kernels.air)
    set(UMAP_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/umap_kernels.metallib)
    add_custom_command(
        OUTPUT ${UMAP_METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${UMAP_METAL_SHADER} -o ${UMAP_METAL_AIR}
        DEPENDS ${UMAP_METAL_SHADER}
        COMMENT "Compiling Metal shaders (umap_kernels.metal -> .air)"
        VERBATIM)
    add_custom_command(
        OUTPUT ${UMAP_METALLIB}
        COMMAND xcrun -sdk macosx metallib ${UMAP_METAL_AIR} -o ${UMAP_METALLIB}
        DEPENDS ${UMAP_METAL_AIR}
        COMMENT "Linking Metal library (umap_kernels.metallib)"
        VERBATIM)
    add_custom_target(umap_metallib ALL DEPENDS ${UMAP_METALLIB})

    target_sources(umapcpp PRIVATE
        src/metal/metal_context.cpp
        src/metal/gpu_knn.cpp
        src/metal/gpu_layout.cpp)
    add_dependencies(umapcpp umap_metallib)
    # metal-cpp (vendored, Apple's C++ Metal API): SYSTEM to keep -Wpedantic quiet.
    target_include_directories(umapcpp SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp)
    target_compile_definitions(umapcpp PUBLIC UMAP_ENABLE_METAL=1)
    # Default metallib location for this build tree; runtime override via the
    # UMAP_METALLIB_PATH environment variable (installation story: Phase 4).
    target_compile_definitions(umapcpp PRIVATE
        UMAP_DEFAULT_METALLIB="${UMAP_METALLIB}")
    target_link_libraries(umapcpp PUBLIC
        "-framework Metal" "-framework Foundation" "-framework QuartzCore")
endif()
# ----------------------------------------------------------------------------

# --- inverse-update additions -----------------------------------------------
# scipy.spatial.Delaunay port for inverse_transform: the exact qhull sources
# scipy 1.18.1 vendors (qhull_r 8.0.2 reentrant, subprojects/qhull_r with
# scipy's poly2rc.patch pre-applied) as a static C library, plus the
# _qhull.pyx port in src/delaunay.cpp. CPU-only builds are unaffected (pure C,
# no external dependencies).
enable_language(C)
add_library(qhull_r STATIC
    third_party/qhull_r/libqhull_r/geom2_r.c
    third_party/qhull_r/libqhull_r/geom_r.c
    third_party/qhull_r/libqhull_r/global_r.c
    third_party/qhull_r/libqhull_r/io_r.c
    third_party/qhull_r/libqhull_r/libqhull_r.c
    third_party/qhull_r/libqhull_r/mem_r.c
    third_party/qhull_r/libqhull_r/merge_r.c
    third_party/qhull_r/libqhull_r/poly2_r.c
    third_party/qhull_r/libqhull_r/poly_r.c
    third_party/qhull_r/libqhull_r/qset_r.c
    third_party/qhull_r/libqhull_r/random_r.c
    third_party/qhull_r/libqhull_r/rboxlib_r.c
    third_party/qhull_r/libqhull_r/stat_r.c
    third_party/qhull_r/libqhull_r/user_r.c
    third_party/qhull_r/libqhull_r/usermem_r.c
    third_party/qhull_r/libqhull_r/userprintf_r.c
    third_party/qhull_r/libqhull_r/userprintf_rbox_r.c)
# phase4: BUILD_INTERFACE-wrapped for the export set (no public umap header
# includes qhull; consumers only need to link the archive).
target_include_directories(qhull_r PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/qhull_r>)
target_compile_options(qhull_r PRIVATE -O2 -w)  # vendored code: warnings off
target_sources(umapcpp PRIVATE src/delaunay.cpp)
target_link_libraries(umapcpp PUBLIC qhull_r)
# ----------------------------------------------------------------------------

enable_testing()

add_executable(umap_tests
    tests/test_main.cpp
    tests/test_distances.cpp
    tests/test_rng.cpp
    tests/test_sparse_matrix.cpp
    tests/test_fuzzy_simplicial_set.cpp
    tests/test_find_ab_params.cpp
)
target_link_libraries(umap_tests PRIVATE umapcpp)
target_include_directories(umap_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(umap_tests PRIVATE -O2 -Wall -Wextra)

# --- layout-sgd additions (branch layout-sgd) -------------------------------
target_sources(umap_tests PRIVATE tests/test_layout.cpp)
# ----------------------------------------------------------------------------

# --- metrics-full additions -------------------------------------------------
target_sources(umap_tests PRIVATE
    tests/test_metrics_full.cpp
    tests/test_layout_generic.cpp)
# ----------------------------------------------------------------------------

# --- nn-descent additions ---------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_nn_descent.cpp)
# ----------------------------------------------------------------------------

# --- query-path additions ---------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_query.cpp)
# ----------------------------------------------------------------------------

# --- sparse-input additions -------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_sparse.cpp)
# ----------------------------------------------------------------------------

# --- sparse-transform additions ---------------------------------------------
target_sources(umap_tests PRIVATE tests/test_sparse_query.cpp)
# ----------------------------------------------------------------------------

# --- sparse-metrics additions -----------------------------------------------
target_sources(umap_tests PRIVATE tests/test_sparse_metrics_full.cpp)
# ----------------------------------------------------------------------------

# --- spectral-init additions ------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_spectral.cpp)
# ----------------------------------------------------------------------------

# --- tswspectral additions ---------------------------------------------------
# Port of test_spectral.py (tswspectral_layout + LOBPCG); goldens from
# scripts/gen_golden_tsw.py -> tests/golden_tsw.hpp.
target_sources(umap_tests PRIVATE tests/test_tswspectral.cpp)
# ----------------------------------------------------------------------------

# --- umap-class additions ---------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_umap.cpp)
# ----------------------------------------------------------------------------

# --- feature-bundle additions ------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_umap_features.cpp)
# ----------------------------------------------------------------------------

# --- densmap additions ------------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_densmap.cpp)
# ----------------------------------------------------------------------------

# --- supervised-sparse additions ---------------------------------------------
target_sources(umap_tests PRIVATE
    tests/test_supervised.cpp
    tests/test_sparse_class.cpp)
# ----------------------------------------------------------------------------

# --- inverse-update additions ------------------------------------------------
target_sources(umap_tests PRIVATE
    tests/test_inverse.cpp
    tests/test_update.cpp)
# ----------------------------------------------------------------------------

# --- composite additions -----------------------------------------------------
# Port of test_composite_models.py (UMAP operator* / + / -); goldens from
# scripts/gen_golden_composite.py -> tests/golden_composite.hpp.
target_sources(umap_tests PRIVATE tests/test_composite_models.cpp)
# ----------------------------------------------------------------------------

# --- aligned-umap additions --------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_aligned.cpp)
# ----------------------------------------------------------------------------

# --- metal-gpu additions ----------------------------------------------------
if(UMAP_ENABLE_METAL)
    target_sources(umap_tests PRIVATE tests/test_metal.cpp)
endif()
# ----------------------------------------------------------------------------

# --- phase4 additions --------------------------------------------------------
# Model persistence round trips + C API coverage (c_api_smoke.c is compiled
# as C, proving umap_c.h is C-consumable).
target_sources(umap_tests PRIVATE
    tests/test_model_io.cpp
    tests/test_c_api.cpp
    tests/c_api_smoke.c)
# ------------------------------------------------------------------------------

# --- tests-port additions ---------------------------------------------------
# Port of the upstream test suite (reference/umap/umap/tests) to doctest; see
# tests/PORTING_NOTES.md. Fixtures generated by scripts/gen_upstream_fixtures.py.
target_sources(umap_tests PRIVATE
    tests/test_umap_metrics.cpp
    tests/test_umap_nn.cpp
    tests/test_umap_trustworthiness.cpp
    tests/test_umap_on_iris.cpp
    tests/test_umap_ops.cpp
    tests/test_umap_repeated_data.cpp
    tests/test_data_input.cpp
    tests/test_umap_validation_params.cpp
    tests/test_umap_get_feature_names_out.cpp
    tests/test_chunked_parallel_spatial_metric.cpp
    tests/test_umap_grads.cpp)
# ----------------------------------------------------------------------------

add_test(NAME umap_tests COMMAND umap_tests)

# --- phase4 additions ---------------------------------------------------------
# CLI executable, install rules and the umapcpp find_package() config.

add_executable(umap apps/umap_cli.cpp)
target_link_libraries(umap PRIVATE umapcpp)
target_compile_options(umap PRIVATE -O2 -Wall -Wextra)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Static archives + CLI. qhull_r is a PUBLIC link dependency of the static
# umapcpp archive, so consumers need both .a files.
install(TARGETS umapcpp qhull_r umap
    EXPORT umapcpp-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/umap
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

install(EXPORT umapcpp-targets
    NAMESPACE umapcpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/umapcpp)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/umapcpp-config-version.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/umapcpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/umapcpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/umapcpp)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/umapcpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/umapcpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/umapcpp)

if(UMAP_ENABLE_METAL)
    # Install the compiled shaders next to the libraries. At runtime the
    # loader tries, in order: $UMAP_METALLIB_PATH, the build-tree path baked
    # via UMAP_DEFAULT_METALLIB, then <dir of the executable/dylib containing
    # libumapcpp>/umap_kernels.metallib and
    # <dir>/../lib/umapcpp/umap_kernels.metallib (dladdr; covers
    # bin/umap -> lib/umapcpp/ in an install tree). See
    # src/metal/metal_context.cpp.
    install(FILES ${UMAP_METALLIB} DESTINATION ${CMAKE_INSTALL_LIBDIR}/umapcpp)
endif()
# ------------------------------------------------------------------------------
