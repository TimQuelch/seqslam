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
                          std::size_t maxTileSize,
                          std::size_t maxNPerThread) {
        for (auto tileSize = 1u; tileSize <= maxTileSize; tileSize++) {
            for (auto nPerThread = 1u; nPerThread <= maxNPerThread; nPerThread++) {
                if (opencl::isValidParameters(nImages, nPix, tileSize, nPerThread)) {
                    b->Args({tileSize, nPerThread});
                }
            }
        }
    }

    auto loadImages(std::string_view path) {
        auto const dataDir = std::filesystem::path{path};
        auto referenceImages =
            convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
        auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

        assert(referenceImages.size() > 0);
        assert(queryImages.size() > 0);
        assert(referenceImages[0].rows() == queryImages[0].rows());
        assert(referenceImages[0].cols() == queryImages[0].cols());
        auto const nRows = queryImages[0].rows();
        auto const nCols = queryImages[0].cols();

        return std::tuple{std::move(referenceImages), std::move(queryImages), nRows, nCols};
    }
} // namespace

void cpuDifferenceMatrix(benchmark::State& state) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(smallImagesDir);

    for (auto _ : state) {
        auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages, state.range(0));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
// BENCHMARK(cpuDifferenceMatrix)
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(4)
//    ->Range(1, 256)
//    ->MinTime(1);

void gpuDifferenceMatrix(benchmark::State& state) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(smallImagesDir);

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, state.range(0));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
// BENCHMARK(gpuDifferenceMatrix)
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(2)
//    ->Range(1, 4)
//    ->MinTime(1);

void gpuDifferenceMatrixNoContext(benchmark::State& state) {
    auto const [referenceImages, queryImages, nRows, nCols] = loadImages(smallImagesDir);

    auto context = opencl::createDiffMxContext();

    for (auto _ : state) {
        auto diffMatrix =
            opencl::generateDiffMx(context, referenceImages, queryImages, state.range(0));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
// BENCHMARK(gpuDifferenceMatrixNoContext)
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(2)
//    ->Range(1, 4)
//    ->MinTime(1);

void gpuDifferenceMatrixNoCopy(benchmark::State& state,
                               std::string const& kernel,
                               std::string_view dir) {
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
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, best, "diffMx", largeImagesDir)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(10)
    ->Apply([](auto b) { return gpuBenchmarkArgs(b, nImages, largeSize, 4, 32); });
// BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, bestNoUnroll, "diffMxNoUnroll")
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(2)
//    ->DenseRange(1, 6)
//    ->MinTime(1);
// BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, stridedIndex, "diffMxStridedIndex")
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(2)
//    ->Range(1, 4)
//    ->MinTime(1);
// BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, serialReduce, "diffMxSerialSave")
//    ->Unit(benchmark::kMillisecond)
//    ->RangeMultiplier(2)
//    ->Range(1, 4)
//    ->MinTime(1);

BENCHMARK_MAIN();
