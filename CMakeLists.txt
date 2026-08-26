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

add_test(NAME umap_tests COMMAND umap_tests)
