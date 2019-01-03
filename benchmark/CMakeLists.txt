cmake_minimum_required(VERSION 3.10)
project(seqslam)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)

add_executable(diffmx-benchmark diffmx-benchmark.cpp)
target_link_libraries(diffmx-benchmark PRIVATE seqslam benchmark::benchmark)

add_executable(enhance-benchmark enhance-benchmark.cpp)
target_link_libraries(enhance-benchmark PRIVATE seqslam benchmark::benchmark)

add_executable(search-benchmark search-benchmark.cpp)
target_link_libraries(search-benchmark PRIVATE seqslam benchmark::benchmark)

add_executable(sad-benchmark sad-benchmark.cpp)
target_compile_features(sad-benchmark PRIVATE cxx_std_17)
target_link_libraries(sad-benchmark PRIVATE benchmark::benchmark Eigen3::Eigen)

add_executable(eigen-benchmark eigen-benchmark.cpp)
target_compile_features(eigen-benchmark PRIVATE cxx_std_17)
target_link_libraries(eigen-benchmark PRIVATE benchmark::benchmark Eigen3::Eigen)
