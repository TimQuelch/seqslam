#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

constexpr auto smallImagesDir = "datasets/nordland_trimmed_resized"sv;
constexpr auto largeImagesDir = "datasets/nordland_trimmed"sv;
constexpr auto smallSize = 16u * 32u;
constexpr auto largeSize = 32u * 64u;
constexpr auto nImages = 3576u;

namespace {
    void gpuBenchmarkArgs(benchmark::internal::Benchmark* b,
                          std::size_t nPix,
                          std::pair<std::size_t, std::size_t> tileSizeRRange,
                          std::pair<std::size_t, std::size_t> tileSizeQRange,
                          std::pair<std::size_t, std::size_t> nPixPerThreadRange) {
        for (auto tileSizeR = tileSizeRRange.first; tileSizeR <= tileSizeRRange.second;
             tileSizeR++) {
            for (auto tileSizeQ = tileSizeQRange.first; tileSizeQ <= tileSizeQRange.second;
                 tileSizeQ++) {
                for (auto nPixPerThread = nPixPerThreadRange.first;
                     nPixPerThread <= nPixPerThreadRange.second;
                     nPixPerThread++) {
                    if (opencl::diffmxcalc::isValidParameters(
                            nImages, nImages, nPix, tileSizeR, tileSizeQ, nPixPerThread)) {
                        b->Args({static_cast<long>(tileSizeR),
                                 static_cast<long>(tileSizeQ),
                                 static_cast<long>(nPixPerThread)});
                    }
                }
            }
        }
    }

    void cpuBenchmarkArgs(benchmark::internal::Benchmark* b) {
        auto args = std::vector<long>{};
        args.push_back(1);
        args.push_back(2);
        for (auto i = 4u; i < 16; i += 4) {
            args.push_back(i);
        }
        for (auto i = 16u; i < 64; i += 16) {
            args.push_back(i);
        }
        for (auto i = 64u; i < 256; i += 64) {
            args.push_back(i);
        }
        for (auto i : args) {
            for (auto j : args) {
                b->Args({i, j});
            }
        }
    }

    auto loadImages(std::string_view path) {
        auto const dataDir = std::filesystem::path{path};
        auto referenceImages =
            convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
        auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

        assert(!referenceImages.empty());
        assert(!queryImages.empty());
        assert(referenceImages[0].rows() == queryImages[0].rows());
        assert(referenceImages[0].cols() == queryImages[0].cols());
        auto const nRows = queryImages[0].rows();
        auto const nCols = queryImages[0].cols();

        return std::tuple{std::move(referenceImages), std::move(queryImages), nRows, nCols};
    }
} // namespace

void cpuDifferenceMatrix(benchmark::State& state, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    for (auto _ : state) {
        auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages, state.range(0));
        benchmark::DoNotOptimize(diffMatrix.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.iterations() * (nRows * nCols * 2 + 1) * referenceImages.size() *
                            queryImages.size() * sizeof(PixType));
}
BENCHMARK_CAPTURE(cpuDifferenceMatrix, small, smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cpuBenchmarkArgs);
BENCHMARK_CAPTURE(cpuDifferenceMatrix, large, largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cpuBenchmarkArgs);

void gpuDifferenceMatrixWithCopyAndContext(benchmark::State& state, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(
            referenceImages, queryImages, state.range(0), state.range(1), state.range(2));
        benchmark::DoNotOptimize(diffMatrix.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.iterations() * (nRows * nCols * 2 + 1) * referenceImages.size() *
                            queryImages.size() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopyAndContext, small, smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopyAndContext, large, largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, largeSize, {1, 256}, {1, 256}, {1, 256});
    });

void gpuDifferenceMatrixWithCopy(benchmark::State& state, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    auto context = opencl::diffmxcalc::createContext();

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(
            context, referenceImages, queryImages, state.range(0), state.range(1), state.range(2));
        benchmark::DoNotOptimize(diffMatrix.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.iterations() * (nRows * nCols * 2 + 1) * referenceImages.size() *
                            queryImages.size() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopy, small, smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopy, large, largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, largeSize, {1, 256}, {1, 256}, {1, 256});
    });

void gpuDifferenceMatrix(benchmark::State& state, std::string const& kernel, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    auto context = opencl::diffmxcalc::createContext();
    auto bufs = opencl::diffmxcalc::createBuffers(
        context, referenceImages.size(), queryImages.size(), nRows * nCols);
    opencl::diffmxcalc::writeArgs(context,
                                  bufs,
                                  referenceImages,
                                  queryImages,
                                  state.range(0),
                                  state.range(1),
                                  state.range(2),
                                  kernel);

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(context,
                                                 bufs.diffMx,
                                                 referenceImages.size(),
                                                 queryImages.size(),
                                                 nRows * nCols,
                                                 state.range(0),
                                                 state.range(1),
                                                 state.range(2),
                                                 kernel);
        benchmark::DoNotOptimize(diffMatrix.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.iterations() * (nRows * nCols * 2 + 1) * referenceImages.size() *
                            queryImages.size() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrix, small_best, "diffMxNDiffs", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, small_warpreduce, "diffMxUnrolledWarpReduce", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, small_twodiffs, "diffMxTwoDiffs", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix,
                  small_continuousindex,
                  "diffMxContinuousIndex",
                  smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 1});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, small_parallelsave, "diffMxParallelSave", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 1});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, small_naive, "diffMxNaive", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, smallSize, {1, 256}, {1, 256}, {1, 1});
    });

BENCHMARK_CAPTURE(gpuDifferenceMatrix, large_best, "diffMxNDiffs", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, largeSize, {1, 256}, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, large_warpreduce, "diffMxUnrolledWarpReduce", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, largeSize, {1, 256}, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, large_twodiffs, "diffMxTwoDiffs", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, largeSize, {1, 256}, {1, 256}, {2, 2});
    });

BENCHMARK_MAIN();
