# --- phase4 additions ---------------------------------------------------------
# Minimal find_package(umapcpp) consumer. Build against an installed tree:
#
#   cmake --install <build dir> --prefix <prefix>
#   cmake -S examples/consumer -B <tmp> -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build <tmp> && <tmp>/consumer_smoke
#
# scripts/test_install.sh runs this end to end.
cmake_minimum_required(VERSION 3.16)
project(umapcpp_consumer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(umapcpp 1.0 REQUIRED)

add_executable(consumer_smoke main.cpp)
target_link_libraries(consumer_smoke PRIVATE umapcpp::umapcpp)

# The C API is part of the same archive; prove the header is C-consumable
# from an installed tree too.
add_executable(consumer_smoke_c main.c)
target_link_libraries(consumer_smoke_c PRIVATE umapcpp::umapcpp)
