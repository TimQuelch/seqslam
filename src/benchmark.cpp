#include "seqslam.h"

#include <benchmark/benchmark.h>

using namespace seqslam;

auto loadImages() {
    auto const dataDir = std::filesystem::path{"../datasets/nordland_trimmed_resized"};
    auto const referenceImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto const queryImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));
    return std::tuple{std::move(referenceImages), std::move(queryImages)};
}

void cpuDifferenceMatrix(benchmark::State& state) {
    auto const [referenceImages, queryImages] = loadImages();

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
//    ->RangeMultiplier(2)
//    ->Range(1, 256)
//    ->MinTime(1);

void gpuDifferenceMatrix(benchmark::State& state) {
    auto const [referenceImages, queryImages] = loadImages();

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, state.range(0));
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
BENCHMARK(gpuDifferenceMatrix)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->MinTime(1);

void gpuDifferenceMatrixNoContext(benchmark::State& state) {
    auto const [referenceImages, queryImages] = loadImages();

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
BENCHMARK(gpuDifferenceMatrixNoContext)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->MinTime(1);

void gpuDifferenceMatrixNoCopy(benchmark::State& state, std::string const& kernel) {
    auto const [referenceImages, queryImages] = loadImages();

    auto context = opencl::createDiffMxContext();
    auto bufs = opencl::createBuffers(context, referenceImages.size(), queryImages.size());
    opencl::writeArgs(context, bufs, referenceImages, queryImages, state.range(0), kernel);

    for (auto _ : state) {
        auto diffMatrix = opencl::generateDiffMx(context,
                                                 bufs.diffMx,
                                                 referenceImages.size(),
                                                 queryImages.size(),
                                                 state.range(0),
                                                 kernel);
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, best, "diffMx")
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->MinTime(1);
BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, stridedIndex, "diffMxStridedIndex")
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->MinTime(1);
BENCHMARK_CAPTURE(gpuDifferenceMatrixNoCopy, serialReduce, "diffMxSerialSave")
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 4)
    ->MinTime(1);

BENCHMARK_MAIN();
