#!/bin/bash

TIME=5
benchmark/benchmark-diffmx --benchmark_format=json --benchmark_out=benchmarks-diffmx.json --benchmark_min_time=$TIME;
benchmark/benchmark-enhance --benchmark_format=json --benchmark_out=benchmarks-enhance.json --benchmark_min_time=$TIME;
benchmark/benchmark-search --benchmark_format=json --benchmark_out=benchmarks-search.json --benchmark_min_time=$TIME;

