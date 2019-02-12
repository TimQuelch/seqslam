#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

namespace {
    constexpr auto const smallImagesDir = "../datasets/nordland_trimmed_resized"sv;
    constexpr auto const largeImagesDir = "../datasets/nordland_trimmed"sv;

    constexpr auto const diffMxTileSize = 20;
    constexpr auto const enhancementWindowSize = 30;
    constexpr auto const searchVMin = 0.0f;
    constexpr auto const searchVMax = 5.0f;
    constexpr auto const searchSequenceLength = 15u;
    constexpr auto const searchNTraj = 10u;

    auto loadImages(std::string_view path) {
        auto const dataDir = std::filesystem::path{path};
        auto referenceImages =
            convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
        auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

        assert(!referenceImages.empty());
        assert(!queryImages.empty());
        assert(referenceImages[0].rows() == queryImages[0].rows());
        assert(referenceImages[0].cols() == queryImages[0].cols());
        return std::tuple{std::move(referenceImages), std::move(queryImages)};
    }

    auto getDiffMx() {
        static auto const diffMx = [imgs = loadImages(smallImagesDir)]() {
            auto mx = cpu::generateDiffMx(std::get<0>(imgs), std::get<1>(imgs));
            mx.array() -= mx.minCoeff();
            mx /= mx.maxCoeff();
            return mx;
        }();
        return diffMx;
    }

    auto getEnhancedDiffMx() {
        static auto const enhanced = []() {
            return cpu::enhanceDiffMx(getDiffMx(), enhancementWindowSize);
        }();
        return enhanced;
    }
} // namespace

void diffMxGeneration(benchmark::State& state, std::string_view imagePath) {
    auto const [r, q] = loadImages(imagePath);

    for (auto _ : state) {
        auto mx = cpu::generateDiffMx(r, q, diffMxTileSize);
        benchmark::DoNotOptimize(mx.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK_CAPTURE(diffMxGeneration, small, smallImagesDir);
BENCHMARK_CAPTURE(diffMxGeneration, large, largeImagesDir);

void diffMxEnhancement(benchmark::State& state) {
    auto const mx = getDiffMx();

    for (auto _ : state) {
        auto enhanced = cpu::enhanceDiffMx(mx, enhancementWindowSize);
        benchmark::DoNotOptimize(enhanced.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(diffMxEnhancement);

void sequenceSearch(benchmark::State& state) {
    auto const enhanced = getEnhancedDiffMx();

    for (auto _ : state) {
        auto searched = cpu::sequenceSearch(
            enhanced, searchSequenceLength, searchVMin, searchVMax, searchNTraj);
        benchmark::DoNotOptimize(searched.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(sequenceSearch);

void cpuWhole(benchmark::State& state, std::string_view imagePath) {
    auto const [r, q] = loadImages(imagePath);

    for (auto _ : state) {
        auto mx = cpu::generateDiffMx(r, q, diffMxTileSize);
        auto enhanced = cpu::enhanceDiffMx(mx, enhancementWindowSize);
        auto searched = cpu::sequenceSearch(
            enhanced, searchSequenceLength, searchVMin, searchVMax, searchNTraj);
        benchmark::DoNotOptimize(searched.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK_CAPTURE(cpuWhole, small, smallImagesDir);
BENCHMARK_CAPTURE(cpuWhole, large, largeImagesDir);

void gpuWhole(benchmark::State& state, std::string_view imagePath) {
    auto const [r, q] = loadImages(imagePath);

    for (auto _ : state) {
        auto mx = opencl::generateDiffMx(r, q, 4, 8);
        auto enhanced = opencl::enhanceDiffMx(mx, enhancementWindowSize, 8);
        auto searched = opencl::sequenceSearch(enhanced, searchSequenceLength, searchVMin, searchVMax, searchNTraj, 6);
        benchmark::DoNotOptimize(searched.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK_CAPTURE(gpuWhole, small, smallImagesDir);
BENCHMARK_CAPTURE(gpuWhole, large, largeImagesDir);

BENCHMARK_MAIN();
