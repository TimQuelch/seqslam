cmake_minimum_required(VERSION 3.10)
project(seqslam)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)

add_executable(benchmark-diffmx benchmark-diffmx.cpp)
target_link_libraries(benchmark-diffmx PRIVATE seqslam benchmark::benchmark)

add_executable(benchmark-enhance benchmark-enhance.cpp)
target_link_libraries(benchmark-enhance PRIVATE seqslam benchmark::benchmark)

add_executable(benchmark-search benchmark-search.cpp)
target_link_libraries(benchmark-search PRIVATE seqslam benchmark::benchmark)

add_executable(benchmark-seqslam benchmark-seqslam.cpp)
target_link_libraries(benchmark-seqslam PRIVATE seqslam benchmark::benchmark)

add_executable(benchmark-sad benchmark-sad.cpp)
target_compile_features(benchmark-sad PRIVATE cxx_std_17)
target_link_libraries(benchmark-sad PRIVATE benchmark::benchmark Eigen3::Eigen)

add_executable(benchmark-eigen benchmark-eigen.cpp)
target_compile_features(benchmark-eigen PRIVATE cxx_std_17)
target_link_libraries(benchmark-eigen PRIVATE benchmark::benchmark Eigen3::Eigen)
