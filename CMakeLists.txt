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

# --- nn-descent additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/nn_descent.cpp)
# ----------------------------------------------------------------------------

# --- query-path additions ---------------------------------------------------
target_sources(umapcpp PRIVATE src/nn_query.cpp)
# ----------------------------------------------------------------------------

# --- sparse-input additions -------------------------------------------------
target_sources(umapcpp PRIVATE src/sparse_distances.cpp src/sparse_nn_descent.cpp)
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

# --- nn-descent additions ---------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_nn_descent.cpp)
# ----------------------------------------------------------------------------

# --- query-path additions ---------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_query.cpp)
# ----------------------------------------------------------------------------

# --- sparse-input additions -------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_sparse.cpp)
# ----------------------------------------------------------------------------

# --- spectral-init additions ------------------------------------------------
target_sources(umap_tests PRIVATE tests/test_spectral.cpp)
# ----------------------------------------------------------------------------

add_test(NAME umap_tests COMMAND umap_tests)
