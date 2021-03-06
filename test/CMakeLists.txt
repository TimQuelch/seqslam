cmake_minimum_required(VERSION 3.10)
project(seqslam)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)

enable_testing()

find_package(Catch2 REQUIRED)
include(Catch)

add_library(test-main OBJECT test-main.cpp)
target_link_libraries(test-main PUBLIC Catch2::Catch2)

add_executable(test-trajectory test-trajectory.cpp)
target_link_libraries(test-trajectory PRIVATE test-main seqslam)
catch_discover_tests(test-trajectory)

add_executable(test-validate-diffmx test-validate-diffmx.cpp)
target_link_libraries(test-validate-diffmx PRIVATE test-main seqslam fmt::fmt)
catch_discover_tests(test-validate-diffmx)

add_executable(test-validate-enhance test-validate-enhance.cpp)
target_link_libraries(test-validate-enhance PRIVATE test-main seqslam fmt::fmt)
catch_discover_tests(test-validate-enhance)

add_executable(test-validate-search test-validate-search.cpp)
target_link_libraries(test-validate-search PRIVATE test-main seqslam fmt::fmt)
catch_discover_tests(test-validate-search)
