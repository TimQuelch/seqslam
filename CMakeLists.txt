cmake_minimum_required(VERSION 3.10)

project(seqslam-benchmark VERSION 0.1.0)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(benchmark REQUIRED)

add_library(seqslam src/seqslam.cpp)

target_compile_features(seqslam PUBLIC cxx_std_17)
target_compile_options(seqslam PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(seqslam
    PUBLIC
    Eigen3::Eigen
    PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    stdc++fs)

add_executable(demo src/demo.cpp)
target_compile_options(demo PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(demo PRIVATE seqslam)

add_executable(seqslam-benchmark src/benchmark.cpp)
target_compile_options(seqslam-benchmark PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(seqslam-benchmark PRIVATE seqslam benchmark)
