#include "seqslam-detail.h"
#include "seqslam.h"

#include <cassert>
#include <string_view>

#include <benchmark/benchmark.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

namespace {
    constexpr auto const smallImagesDir = "datasets/nordland_trimmed_resized"sv;
    constexpr auto const nImages = 3576u;
    constexpr auto const smallRange = std::pair{0.8f, 1.2f};
    constexpr auto const largeRange = std::pair{0.0f, 5.0f};

    auto cpuParameters(benchmark::internal::Benchmark* b) {
        for (auto sequenceLength = 3u; sequenceLength <= 30u; sequenceLength += 3u) {
            for (auto nTrajectories = 1u; nTrajectories <= 11u; nTrajectories += 2u) {
                b->Args({static_cast<long>(sequenceLength), static_cast<long>(nTrajectories)});
            }
        }
    }

    auto gpuParameters(benchmark::internal::Benchmark* b) {
        for (auto sequenceLength = 3u; sequenceLength <= 30u; sequenceLength += 3u) {
            for (auto nTrajectories = 1u; nTrajectories <= 11u; nTrajectories += 2u) {
                for (auto nPixPerThread = 1u; nPixPerThread <= 50u; nPixPerThread++) {
                    if (opencl::seqsearch::isValidParameters(nImages, nPixPerThread)) {
                        b->Args({static_cast<long>(sequenceLength),
                                 static_cast<long>(nTrajectories),
                                 static_cast<long>(nPixPerThread)});
                    }
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
        auto const nRows = queryImages[0].rows();
        auto const nCols = queryImages[0].cols();

        return std::tuple{std::move(referenceImages), std::move(queryImages), nRows, nCols};
    }

    auto getDiffMx() {
        static auto const diffMx = [imgs = loadImages(smallImagesDir)]() {
            auto mx =
                cpu::enhanceDiffMx(cpu::generateDiffMx(std::get<0>(imgs), std::get<1>(imgs)), 10);
            mx.array() -= mx.minCoeff();
            mx /= mx.maxCoeff();
            return mx;
        }();
        return diffMx;
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

void cpuSearch(benchmark::State& state, std::pair<float, float> vRange) {
    Mx const mx = getDiffMx();
    auto const sequenceLength = state.range(0);
    auto const nTrajectories = state.range(1);

    for (auto _ : state) {
        auto sequences =
            cpu::sequenceSearch(mx, sequenceLength, vRange.first, vRange.second, nTrajectories);
        benchmark::DoNotOptimize(sequences.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(mx.rows() * mx.cols() * state.iterations());
    state.SetBytesProcessed(state.items_processed() * (sequenceLength * nTrajectories + 1) *
                            sizeof(PixType));
}
BENCHMARK_CAPTURE(cpuSearch, small, smallRange)->Apply(cpuParameters);
BENCHMARK_CAPTURE(cpuSearch, large, largeRange)->Apply(cpuParameters);

void gpuSearch(benchmark::State& state, std::pair<float, float> vRange) {
    Mx const mx = getDiffMx();
    auto const sequenceLength = state.range(0);
    auto const nTrajectories = state.range(1);
    auto const nPixPerThread = state.range(2);

    auto context = opencl::seqsearch::createContext();
    auto bufs = opencl::seqsearch::createBuffers(
        context, sequenceLength, nTrajectories, mx.rows(), mx.cols());
    opencl::seqsearch::writeArgs(context,
                                 bufs,
                                 mx,
                                 sequenceLength,
                                 vRange.first,
                                 vRange.second,
                                 nTrajectories,
                                 nPixPerThread,
                                 opencl::seqsearch::defaultKernel);

    for (auto _ : state) {
        auto sequences =
            opencl::sequenceSearch(context, bufs.out, mx.rows(), mx.cols(), nPixPerThread);
        benchmark::DoNotOptimize(sequences.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(mx.rows() * mx.cols() * state.iterations());
    state.SetBytesProcessed(state.items_processed() * (sequenceLength * nTrajectories + 1) *
                            sizeof(PixType));
}
BENCHMARK_CAPTURE(gpuSearch, small, smallRange)->Apply(gpuParameters);
BENCHMARK_CAPTURE(gpuSearch, large, largeRange)->Apply(gpuParameters);

BENCHMARK_MAIN();
