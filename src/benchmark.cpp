#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

constexpr auto smallImagesDir = "../datasets/nordland_trimmed_resized"sv;
constexpr auto largeImagesDir = "../datasets/nordland_trimmed"sv;
constexpr auto smallSize = 16u * 32u;
constexpr auto largeSize = 32u * 64u;
constexpr auto nImages = 3576u;

namespace {
    void gpuBenchmarkArgs(benchmark::internal::Benchmark* b,
                          std::size_t nImages,
                          std::size_t nPix,
                          std::pair<std::size_t, std::size_t> tileSizeRange,
                          std::pair<std::size_t, std::size_t> nPixPerThreadRange) {
        for (auto tileSize = tileSizeRange.first; tileSize <= tileSizeRange.second; tileSize++) {
            for (auto nPixPerThread = nPixPerThreadRange.first;
                 nPixPerThread <= nPixPerThreadRange.second;
                 nPixPerThread++) {
                if (opencl::isValidParameters(nImages, nPix, tileSize, nPixPerThread)) {
                    b->Args({static_cast<long>(tileSize), static_cast<long>(nPixPerThread)});
                }
            }
        }
    }

    void cpuBenchmarkArgs(benchmark::internal::Benchmark* b) {
        b->Args({1});
        b->Args({2});
        for (auto i = 4u; i < 32; i += 4) {
            b->Args({static_cast<long>(i)});
        }
        for (auto i = 32u; i < 128; i += 8) {
            b->Args({static_cast<long>(i)});
        }
        for (auto i = 128u; i < 256; i += 32) {
            b->Args({static_cast<long>(i)});
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
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.items_processed() * sizeof(PixType));
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
        auto diffMatrix =
            opencl::generateDiffMx(referenceImages, queryImages, state.range(0), state.range(1));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.items_processed() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopyAndContext, small, smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopyAndContext, large, largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {1, 256});
    });

void gpuDifferenceMatrixWithCopy(benchmark::State& state, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    auto context = opencl::createDiffMxContext();

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(
            context, referenceImages, queryImages, state.range(0), state.range(1));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.items_processed() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopy, small, smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrixWithCopy, large, largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {1, 256});
    });

void gpuDifferenceMatrix(benchmark::State& state, std::string const& kernel, std::string_view dir) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(dir);

    auto context = opencl::createDiffMxContext();
    auto bufs =
        opencl::createBuffers(context, referenceImages.size(), queryImages.size(), nRows * nCols);
    opencl::writeArgs(
        context, bufs, referenceImages, queryImages, state.range(0), state.range(1), kernel);

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(context,
                                                 bufs.diffMx,
                                                 referenceImages.size(),
                                                 queryImages.size(),
                                                 nRows * nCols,
                                                 state.range(0),
                                                 state.range(1),
                                                 kernel);
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
    state.SetBytesProcessed(state.items_processed() * sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuDifferenceMatrix, best, "diffMxPreload", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, ndiffs, "diffMxNDiffs", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, warpreduce, "diffMxUnrolledWarpReduce", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, twodiffs, "diffMxTwoDiffs", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, continuousindex, "diffMxContinuousIndex", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 1});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, parallelsave, "diffMxParallelSave", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 1});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, naive, "diffMxNaive", smallImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, smallSize, {1, 256}, {1, 1});
    });

BENCHMARK_CAPTURE(gpuDifferenceMatrix, largebest, "diffMxPreload", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, largendiffs, "diffMxNDiffs", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {1, 256});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, largewarpreduce, "diffMxUnrolledWarpReduce", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {2, 2});
    });
BENCHMARK_CAPTURE(gpuDifferenceMatrix, largetwodiffs, "diffMxTwoDiffs", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->Apply([](auto b) {
        return gpuBenchmarkArgs(b, nImages, largeSize, {1, 256}, {2, 2});
    });

BENCHMARK_MAIN();
