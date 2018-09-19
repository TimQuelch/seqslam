#include <iostream>
#include <random>

#include <type_traits>

#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

constexpr auto rows = 32;
constexpr auto cols = 64;
constexpr auto nVecImages = 32768;
constexpr auto nSqrImages = 4096;

template <typename T>
using Mx = Eigen::Matrix<T, rows, cols>;

template <typename T>
using FlatMx = Eigen::Matrix<T, rows * cols, 1>;

template <typename T>
using VecMx = std::vector<Mx<T>, Eigen::aligned_allocator<Mx<T>>>;

template <typename T>
using VecFlatMx = std::vector<FlatMx<T>, Eigen::aligned_allocator<FlatMx<T>>>;

template <typename T>
auto dist = []() {
    if constexpr (std::is_integral_v<T>) {
        return std::uniform_int_distribution<T>{0, 255};
    } else {
        return std::uniform_real_distribution<T>{0, 255};
    }
}();

auto rng = std::mt19937{42};

template <typename T>
auto randMx() {
    Mx<T> mx;
    for (auto i = 0u; i < mx.rows(); i++) {
        for (auto j = 0u; j < mx.cols(); j++) {
            mx(i, j) = dist<T>(rng);
        }
    }
    return mx;
}

template <typename T>
auto randFlatMx() {
    FlatMx<T> mx;
    for (auto i = 0u; i < mx.rows(); i++) {
        mx(i) = dist<T>(rng);
    }
    return mx;
}

template <typename T>
auto genVecMx(unsigned n) {
    VecMx<T> v;
    v.reserve(n);
    std::generate_n(std::back_inserter(v), n, randMx<T>);
    return v;
}

template <typename T>
auto genVecFlatMx(unsigned n) {
    VecFlatMx<T> v;
    v.reserve(n);
    std::generate_n(std::back_inserter(v), n, randFlatMx<T>);
    return v;
}

template <typename T>
void sadMx(benchmark::State& state) {
    auto a = randMx<T>();
    auto b = randMx<T>();
    for (auto _ : state) {
        auto d = (a - b).cwiseAbs().sum();
        benchmark::DoNotOptimize(&d);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}
// BENCHMARK_TEMPLATE(sadMx, uint8_t);
// BENCHMARK_TEMPLATE(sadMx, int16_t);
// BENCHMARK_TEMPLATE(sadMx, int32_t);
// BENCHMARK_TEMPLATE(sadMx, float);
// BENCHMARK_TEMPLATE(sadMx, double);

template <typename T>
void sadFlatMx(benchmark::State& state) {
    auto a = randFlatMx<uint8_t>();
    auto b = randFlatMx<uint8_t>();
    for (auto _ : state) {
        auto d = (a - b).cwiseAbs().sum();
        benchmark::DoNotOptimize(&d);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}
// BENCHMARK_TEMPLATE(sadFlatMx, uint8_t);
// BENCHMARK_TEMPLATE(sadFlatMx, int16_t);
// BENCHMARK_TEMPLATE(sadFlatMx, int32_t);
// BENCHMARK_TEMPLATE(sadFlatMx, float);
// BENCHMARK_TEMPLATE(sadFlatMx, double);

template <typename T>
void vecSadMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecMx<T>(n);
    auto b = genVecMx<T>(n);
    for (auto _ : state) {
        for (auto i = 0u; i < n; i++) {
            auto d = (a[i] - b[i]).cwiseAbs().sum();
            benchmark::DoNotOptimize(&d);
        }
    }
    state.SetItemsProcessed(state.iterations() * n * rows * cols);
}
// BENCHMARK_TEMPLATE(vecSadMx, uint8_t)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadMx, int16_t)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadMx, int32_t)->Arg(nVecImages);
BENCHMARK_TEMPLATE(vecSadMx, float)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadMx, double)->Arg(nVecImages);

template <typename T>
void vecSadFlatMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecFlatMx<T>(n);
    auto b = genVecFlatMx<T>(n);
    for (auto _ : state) {
        for (auto i = 0u; i < n; i++) {
            auto d = (a[i] - b[i]).cwiseAbs().sum();
            benchmark::DoNotOptimize(&d);
        }
    }
    state.SetItemsProcessed(state.iterations() * n * rows * cols);
}
// BENCHMARK_TEMPLATE(vecSadFlatMx, uint8_t)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadFlatMx, int16_t)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadFlatMx, int32_t)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadFlatMx, float)->Arg(nVecImages);
// BENCHMARK_TEMPLATE(vecSadFlatMx, double)->Arg(nVecImages);

template <typename T>
void sqrSadMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecMx<T>(n);
    auto b = genVecMx<T>(n);
    for (auto _ : state) {
        for (auto i = 0u; i < n; i++) {
            for (auto j = 0u; j < n; j++) {
                auto d = (a[i] - b[j]).cwiseAbs().sum();
                benchmark::DoNotOptimize(&d);
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * n * n * rows * cols);
}
// BENCHMARK_TEMPLATE(sqrSadMx, uint8_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadMx, int16_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadMx, int32_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadMx, float)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadMx, double)->Arg(nSqrImages);

template <typename T>
void sqrSadFlatMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecFlatMx<T>(n);
    auto b = genVecFlatMx<T>(n);
    for (auto _ : state) {
        for (auto i = 0u; i < n; i++) {
            for (auto j = 0u; j < n; j++) {
                auto d = (a[i] - b[j]).cwiseAbs().sum();
                benchmark::DoNotOptimize(&d);
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * n * n * rows * cols);
}
// BENCHMARK_TEMPLATE(sqrSadFlatMx, uint8_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadFlatMx, int16_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadFlatMx, int32_t)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrSadFlatMx, float)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrSadFlatMx, double)->Arg(nSqrImages);

template <typename T, int TileSize>
void sqrCacheOptimisedSadMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecMx<T>(n);
    auto b = genVecMx<T>(n);

    for (auto _ : state) {
        for (auto tx = 0; tx < n / TileSize; tx++) {
            for (auto ty = 0; ty < n / TileSize; ty++) {
                for (auto i = 0u; i < TileSize; i++) {
                    for (auto j = 0u; j < TileSize; j++) {
                        auto d = (a[tx * TileSize + i] - b[ty * TileSize + j]).cwiseAbs().sum();
                        benchmark::DoNotOptimize(&d);
                    }
                }
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * n * n * rows * cols);
}
// BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, uint8_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, int16_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, int32_t)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 4)->Arg(nSqrImages);
// BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, double)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 8)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 16)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 32)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 64)->Arg(nSqrImages);
BENCHMARK_TEMPLATE(sqrCacheOptimisedSadMx, float, 128)->Arg(nSqrImages);

template <typename T, int TileSize>
void sqrCacheOptimisedThreadSadMx(benchmark::State& state) {
    auto n = state.range(0);
    auto a = genVecMx<T>(n);
    auto b = genVecMx<T>(n);

    for (auto _ : state) {
#pragma omp parallel for
        for (auto tx = 0; tx < n / TileSize; tx++) {
            for (auto ty = 0; ty < n / TileSize; ty++) {
                for (auto i = 0u; i < TileSize; i++) {
                    for (auto j = 0u; j < TileSize; j++) {
                        auto d = (a[tx * TileSize + i] - b[ty * TileSize + j]).cwiseAbs().sum();
                        benchmark::DoNotOptimize(&d);
                    }
                }
            }
        }
    }
    state.SetItemsProcessed(state.iterations() * n * n * rows * cols);
}
BENCHMARK_TEMPLATE(sqrCacheOptimisedThreadSadMx, float, 16)->Arg(nSqrImages);

BENCHMARK_MAIN();
