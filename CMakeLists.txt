cmake_minimum_required(VERSION 3.10)

project(seqslam VERSION 0.1.0)

message("Build type: ${CMAKE_BUILD_TYPE}")

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake OPTIONAL)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(benchmark REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/warnings.cmake)
set(optimisation_flags_release -O3 -march=native)
set(optimisation_flags_debug -O0)
set(debug_flags -g)
add_compile_options(
    ${warnings}
    "$<$<CONFIG:Debug>:${optimisation_flags_debug}>"
    "$<$<CONFIG:Debug>:${debug_flags}>"
    "$<$<CONFIG:Release>:${optimisation_flags_release}>")

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
target_include_directories(
    seqslam
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)
target_compile_features(seqslam PUBLIC cxx_std_17)
target_link_libraries(seqslam
    PUBLIC
    Eigen3::Eigen
    fmt::fmt
    clutils
    opencv_core
    PRIVATE
    utils
    opencv_imgproc
    opencv_highgui
    stdc++fs)

add_library(clutils src/clutils.cpp)
target_include_directories(
    clutils
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)
target_compile_features(clutils PUBLIC cxx_std_17)
target_link_libraries(clutils
    PUBLIC
    OpenCL::OpenCL
    stdc++fs
    PRIVATE
    utils
    fmt::fmt
    nlohmann_json::nlohmann_json)

add_library(utils src/utils.cpp)
target_include_directories(
    utils
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>)
target_compile_features(utils PUBLIC cxx_std_17)
target_link_libraries(utils PUBLIC stdc++fs)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE seqslam fmt::fmt)

configure_file(src/kernels/diff-mx.cl ./kernels/diff-mx.cl COPYONLY)
configure_file(src/kernels/enhancement.cl ./kernels/enhancement.cl COPYONLY)
configure_file(clconfig.json clconfig.json COPYONLY)

enable_testing()
add_subdirectory(test)
add_subdirectory(benchmark)

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
