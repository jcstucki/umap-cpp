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
target_include_directories(umapcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
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
target_include_directories(umapcpp SYSTEM PUBLIC
    ${eigen3_SOURCE_DIR}
    ${spectra_SOURCE_DIR}/include
)
# ----------------------------------------------------------------------------

# --- umap-class additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/umap.cpp)
# ----------------------------------------------------------------------------

# --- supervised-sparse additions ---------------------------------------------
target_sources(umapcpp PRIVATE src/supervised.cpp)
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

# --- spectral-init additions ------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_spectral.cpp)
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

# --- metal-gpu additions ----------------------------------------------------
if(UMAP_ENABLE_METAL)
    target_sources(umap_tests PRIVATE tests/test_metal.cpp)
endif()
# ----------------------------------------------------------------------------

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
