#include "seqslam.h"

#include <benchmark/benchmark.h>

using namespace seqslam;

void cpuDifferenceMatrix(benchmark::State& state) {
    const auto dataDir = std::filesystem::path{"../datasets/nordland_trimmed"};
    const auto referenceImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    const auto queryImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

    for (auto _ : state) {
        auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
        benchmark::DoNotOptimize(diffMatrix.get());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * referenceImages.size() * queryImages.size() *
                            nRows * nCols);
}
BENCHMARK(cpuDifferenceMatrix)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
