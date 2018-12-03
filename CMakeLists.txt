cmake_minimum_required(VERSION 3.10)

project(seqslam VERSION 0.1.0)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake OPTIONAL)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(benchmark REQUIRED)
find_package(fmt REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/warnings.cmake)
set(optimisations -O3 -march=native)
add_compile_options(${warnings} ${optimisations})

option(RUN_CLANG_TIDY "Run clang tidy on C++ code" FALSE)
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy not found.")
else()
    message(STATUS "clang-tidy found.")
    set(
        CLANG_TIDY
        "${CLANG_TIDY_EXE}"
        "-enable-check-profile")
endif()

add_library(seqslam src/seqslam.cpp)
target_compile_features(seqslam PUBLIC cxx_std_17)
target_link_libraries(seqslam
    PUBLIC
    Eigen3::Eigen
    fmt::fmt
    clutils
    opencv_core
    PRIVATE
    opencv_imgproc
    opencv_highgui
    stdc++fs)

add_library(clutils src/clutils.cpp)
target_compile_features(clutils PUBLIC cxx_std_17)
target_link_libraries(clutils PUBLIC OpenCL::OpenCL stdc++fs PRIVATE fmt::fmt)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE seqslam fmt::fmt)

add_executable(seqslam-benchmark src/benchmark.cpp)
target_link_libraries(seqslam-benchmark PRIVATE seqslam benchmark::benchmark)

add_executable(seqslam-test src/test.cpp)
target_link_libraries(seqslam-test PRIVATE seqslam fmt::fmt)

add_executable(sad-benchmark src/sad-benchmark.cpp)
target_compile_features(sad-benchmark PRIVATE cxx_std_17)
target_link_libraries(sad-benchmark PRIVATE benchmark::benchmark Eigen3::Eigen)

configure_file(src/kernels/diff-mx.cl ./kernels/diff-mx.cl COPYONLY)

if(CLANG_TIDY_EXE AND RUN_CLANG_TIDY)
    set_target_properties(
        seqslam
        clutils
        demo
        seqslam-benchmark
        sad-benchmark
        PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY}")
endif()
