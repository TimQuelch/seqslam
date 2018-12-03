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
