#include "seqslam.h"

#include <benchmark/benchmark.h>

using namespace seqslam;

void cpuDifferenceMatrix(benchmark::State& state) {
    auto dataDir = std::filesystem::path{"../datasets/nordland_trimmed"};
    auto referenceImages = convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

    for (auto _ : state) {
        auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    }
}
BENCHMARK(cpuDifferenceMatrix)->Unit(benchmark::kMillisecond);

void cpuDifferenceMatrixEnhancement(benchmark::State& state) {
    auto dataDir = std::filesystem::path{"../datasets/nordland_trimmed"};
    auto referenceImages = convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));
    auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);

    for (auto _ : state) {
        auto enhanced = cpu::enhanceDiffMxContrast(diffMatrix);
    }
}
BENCHMARK(cpuDifferenceMatrixEnhancement)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
