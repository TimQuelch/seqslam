#include <random>
#include <memory>

#include <benchmark/benchmark.h>

#include <Eigen/Dense>

using Type = float;
using Mx = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

auto rng = std::mt19937{42};
auto dist = std::uniform_real_distribution<Type>{0, 1};

auto emptyMx(std::size_t n) -> Mx {
    Mx mx{n, n};
    return mx;
}

auto emptyMxUniquePtr(std::size_t n) -> std::unique_ptr<Mx> {
    auto mx = std::make_unique<Mx>(n, n);
    return mx;
}

auto randMx(std::size_t n) -> Mx {
    Mx mx{n, n};

    for (auto i = 0u; i < mx.rows(); ++i) {
        for (auto j = 0u; j < mx.cols(); ++j) {
            mx(i, j) = dist(rng);
        }
    }

    return mx;
}

auto randMxUniquePtr(std::size_t n) -> std::unique_ptr<Mx> {
    auto mx = std::make_unique<Mx>(n, n);

    for (auto i = 0u; i < mx->rows(); ++i) {
        for (auto j = 0u; j < mx->cols(); ++j) {
            (*mx)(i, j) = dist(rng);
        }
    }

    return mx;
}

void genEmptyMx(benchmark::State& state) {
    auto n = state.range(0);
    for (auto _ : state) {
        Mx mx = emptyMx(n);
        benchmark::DoNotOptimize(mx.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * n * n);
}
BENCHMARK(genEmptyMx)->Range(1 << 4, 1 << 13);

void genRandMx(benchmark::State& state) {
    auto n = state.range(0);
    for (auto _ : state) {
        Mx mx = randMx(n);
        benchmark::DoNotOptimize(mx.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * n * n);
}
BENCHMARK(genRandMx)->Range(1 << 4, 1 << 13);

void genEmptyMxUniquePtr(benchmark::State& state) {
    auto n = state.range(0);
    for (auto _ : state) {
        auto mx = emptyMxUniquePtr(n);
        benchmark::DoNotOptimize(mx->data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * n * n);
}
BENCHMARK(genEmptyMxUniquePtr)->Range(1 << 4, 1 << 13);

void genRandMxUniquePtr(benchmark::State& state) {
    auto n = state.range(0);
    for (auto _ : state) {
        auto mx = randMxUniquePtr(n);
        benchmark::DoNotOptimize(mx->data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * n * n);
}
BENCHMARK(genRandMxUniquePtr)->Range(1 << 4, 1 << 13);

BENCHMARK_MAIN();
