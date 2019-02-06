#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

constexpr auto const nImages = 3576u;

namespace {
    constexpr auto const smallImagesDir = "../datasets/nordland_trimmed_resized"sv;

    auto cpuParameters(benchmark::internal::Benchmark* b) {
        for (auto windowSize = 4u; windowSize <= 512u;
             windowSize = windowSize < 32 ? windowSize * 2u : windowSize + 32u) {
            b->Args({static_cast<long>(windowSize)});
        }
    }

    auto gpuParameters(benchmark::internal::Benchmark* b, std::size_t nReference) {
        for (auto windowSize = 4u; windowSize <= 512u;
             windowSize = windowSize < 32 ? windowSize * 2u : windowSize + 32u) {
            for (auto nPixPerThread = 1u; nPixPerThread <= 50u; nPixPerThread++) {
                if (opencl::diffmxenhance::isValidParameters(nReference, nPixPerThread)) {
                    b->Args({static_cast<long>(windowSize), static_cast<long>(nPixPerThread)});
                }
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
} // namespace

void cpuEnhancement(benchmark::State& state) {
    Mx mx = getDiffMx();
    auto const windowSize = state.range(0);

    for (auto _ : state) {
        auto enhanced = cpu::enhanceDiffMx(mx, windowSize);
        benchmark::DoNotOptimize(enhanced.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(mx.rows() * mx.cols() * state.iterations());
    state.SetBytesProcessed(state.items_processed() * (windowSize + 1) * sizeof(PixType));
}
BENCHMARK(cpuEnhancement)->Apply(cpuParameters);

void gpuEnhancement(benchmark::State& state) {
    Mx mx = getDiffMx();
    auto const windowSize = state.range(0);
    auto const nPixPerThread = state.range(1);

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
    state.SetItemsProcessed(mx.rows() * mx.cols() * state.iterations());
    state.SetBytesProcessed(state.items_processed() * (windowSize + 1) * sizeof(PixType));
}
BENCHMARK(gpuEnhancement)->Apply([](auto b) { return gpuParameters(b, nImages); });

BENCHMARK_MAIN();
