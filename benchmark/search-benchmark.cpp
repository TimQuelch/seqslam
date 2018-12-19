#include "seqslam-detail.h"
#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

[[maybe_unused]] constexpr auto smallImagesDir = "../datasets/nordland_trimmed_resized"sv;
[[maybe_unused]] constexpr auto largeImagesDir = "../datasets/nordland_trimmed"sv;
[[maybe_unused]] constexpr auto smallSize = 16u * 32u;
[[maybe_unused]] constexpr auto largeSize = 32u * 64u;
[[maybe_unused]] constexpr auto nImages = 3576u;

namespace {
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

    auto getDiffMx() {
        static auto const imgs = loadImages(largeImagesDir);
        static auto const mx = cpu::generateDiffMx(std::get<0>(imgs), std::get<1>(imgs));
        return mx;
    }
} // namespace

void queryIndexOffsets(benchmark::State& state) {
    for (auto _ : state) {
        auto qi = detail::calcTrajectoryQueryIndexOffsets(state.range(0));
        benchmark::DoNotOptimize(qi.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(queryIndexOffsets)->RangeMultiplier(2)->Range(1, 1 << 8);

void referenceIndexOffsets(benchmark::State& state) {
    auto qi = detail::calcTrajectoryQueryIndexOffsets(state.range(0));
    for (auto _ : state) {
        auto ri = detail::calcTrajectoryReferenceIndexOffsets(qi, 0, 5, 50);
    }
}
BENCHMARK(referenceIndexOffsets)->RangeMultiplier(2)->Range(1, 1 << 8);

void diffMxEnhancement(benchmark::State& state) {
    Mx mx = getDiffMx();
    auto const windowSize = state.range(0);
    for (auto _ : state) {
        auto enhanced = cpu::enhanceDiffMx(mx, windowSize);
    }
    state.SetBytesProcessed(mx.rows() * mx.cols() * windowSize * state.iterations() *
                            sizeof(PixType));
}
BENCHMARK(diffMxEnhancement)->RangeMultiplier(2)->Range(1 << 2, 1 << 8);

void gpuDiffMxEnhancement(benchmark::State& state) {
    Mx mx = getDiffMx();
    auto const windowSize = state.range(0);
    auto const nPixPerThread = 4u;

    auto context = opencl::diffmxenhance::createContext();
    auto bufs = opencl::diffmxenhance::createBuffers(context, mx.rows(), mx.cols());
    opencl::diffmxenhance::writeArgs(
        context, bufs, mx, windowSize, nPixPerThread, opencl::diffmxenhance::defaultKernel);

    for (auto _ : state) {
        auto enhanced =
            opencl::enhanceDiffMx(context, bufs.out, mx.rows(), mx.cols(), nPixPerThread);
        benchmark::DoNotOptimize(enhanced.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(mx.rows() * mx.cols() * windowSize * state.iterations() *
                            sizeof(PixType));
}
BENCHMARK(gpuDiffMxEnhancement)->RangeMultiplier(2)->Range(1 << 2, 1 << 8);

void sequenceSearch(benchmark::State& state) {
    Mx mx = cpu::enhanceDiffMx(getDiffMx(), 10);
    for (auto _ : state) {
        auto sequences = cpu::sequenceSearch(mx, state.range(0), 0, 5, 1);
    }
    state.SetBytesProcessed(mx.rows() * mx.cols() * state.iterations() * sizeof(PixType));
}
BENCHMARK(sequenceSearch)->RangeMultiplier(2)->Range(1 << 2, 1 << 8);

void sequenceSearchVaryingNTrajectories(benchmark::State& state) {
    Mx mx = cpu::enhanceDiffMx(getDiffMx(), 10);
    for (auto _ : state) {
        auto sequences = cpu::sequenceSearch(mx, 30, 0, 5, state.range(0));
    }
    state.SetBytesProcessed(mx.rows() * mx.cols() * state.iterations() * sizeof(PixType));
}
BENCHMARK(sequenceSearchVaryingNTrajectories)->DenseRange(1, 10);

BENCHMARK_MAIN();
