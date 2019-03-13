#include "seqslam.h"

#include <cassert>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>

#include <fmt/format.h>

#include <catch2/catch.hpp>

using namespace seqslam;
using namespace std::literals::string_view_literals;

constexpr auto const smallImagesDir = "datasets/nordland_trimmed_resized"sv;

struct DiffMxComparison {
    float max;
    float mean;
    float std;
};
auto compareDiffMx(Mx const& one, Mx const& two) {
    Mx const difference = (one - two).cwiseAbs();
    auto const max = difference.maxCoeff();
    auto const mean = difference.mean();
    auto const std = std::sqrt((difference.array() - mean).pow(2).sum() /
                               (difference.rows() * difference.cols() - 1));
    return DiffMxComparison{max, mean, std};
}

auto loadImages(std::string_view path) {
    auto const dataDir = std::filesystem::path{path};
    auto referenceImages = convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto queryImages = convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

    assert(!referenceImages.empty());
    assert(!queryImages.empty());
    assert(referenceImages[0].rows() == queryImages[0].rows());
    assert(referenceImages[0].cols() == queryImages[0].cols());
    return std::tuple{std::move(referenceImages), std::move(queryImages)};
}

auto getDiffMx() {
    static auto const diffMx = [imgs = loadImages(smallImagesDir)]() {
        return cpu::enhanceDiffMx(cpu::generateDiffMx(std::get<0>(imgs), std::get<1>(imgs)), 10);
    }();
    return diffMx;
}

template <typename T>
auto generateRange(std::tuple<T, T, unsigned> range) {
    assert(range.first < range.second);
    auto ret = std::vector<T>{};

    auto const min = static_cast<double>(std::get<0>(range));
    auto const max = static_cast<double>(std::get<1>(range));

    auto const delta = (max - min) / std::get<2>(range);

    for (auto val = min; val <= max; val += delta) {
        ret.push_back(val);
    }

    ret.erase(std::unique(ret.begin(), ret.end()), ret.end());

    return ret;
}

struct Parameters {
    unsigned sequenceLength;
    unsigned nTrajectories;
    float vMin;
    float vMax;
};

auto generateParameters(std::tuple<unsigned, unsigned, unsigned> const& sequenceLengthRange,
                        std::tuple<unsigned, unsigned, unsigned> const& nTrajectoriesRange,
                        std::tuple<float, float, unsigned> const& vMinRange,
                        std::tuple<float, float, unsigned> const& vMaxRange,
                        unsigned nTestsToAccept) {
    auto const sequenceLengthVals = generateRange(sequenceLengthRange);
    auto const nTrajectoriesVals = generateRange(nTrajectoriesRange);
    auto const vMinVals = generateRange(vMinRange);
    auto const vMaxVals = generateRange(vMaxRange);

    auto ret = std::vector<Parameters>{};
    for (auto sequenceLength : sequenceLengthVals) {
        for (auto nTrajectories : nTrajectoriesVals) {
            for (auto vMin : vMinVals) {
                for (auto vMax : vMaxVals) {
                    if (vMin < vMax) {
                        ret.push_back({sequenceLength, nTrajectories, vMin, vMax});
                    }
                }
            }
        }
    }

    if (ret.size() < nTestsToAccept) {
        return ret;
    } else {
        auto rng = std::mt19937{42};
        auto sampledRet = decltype(ret){};
        sampledRet.reserve(nTestsToAccept);
        std::sample(ret.begin(), ret.end(), std::back_inserter(sampledRet), nTestsToAccept, rng);
        return sampledRet;
    }
}

namespace fmt {
    template <>
    struct formatter<DiffMxComparison> {
        template <typename ParseContext>
        constexpr auto parse(ParseContext& context) {
            return context.begin();
        }

        template <typename FormatContext>
        auto format(DiffMxComparison const& c, FormatContext& context) {
            return format_to(context.begin(),
                             "Max diff: {:10}, Mean diff {:10}, Std diff {:10}",
                             c.max,
                             c.mean,
                             c.std);
        }
    };
} // namespace fmt

TEST_CASE("Sequence search results consistent with all parameters", "[seqsearch]") {
    auto const mx = getDiffMx();

    auto const parameters =
        generateParameters({5, 35, 30}, {3, 30, 15}, {0.0, 3.0, 10}, {0.5, 5.0, 10}, 15);

    for (auto const p : parameters) {
        auto results = std::vector<std::pair<Mx, std::string>>{};

        // Compute with CPU
        results.push_back(
            {cpu::sequenceSearch(mx, p.sequenceLength, p.vMin, p.vMax, p.nTrajectories),
             fmt::format(
                 "CPU (sequence length = {}, vMin = {}, vMax = {}, number trajectories = {})",
                 p.sequenceLength,
                 p.vMin,
                 p.vMax,
                 p.nTrajectories)});

        // Compute with GPU
        for (auto nPixPerThread = 1u; nPixPerThread <= 50u; nPixPerThread++) {
            if (opencl::seqsearch::isValidParameters(mx.rows(), nPixPerThread)) {
                results.push_back(
                    {opencl::sequenceSearch(mx, p.sequenceLength, p.vMin, p.vMax, p.nTrajectories),
                     fmt::format("GPU (sequence length = {}, vMin = {}, vMax = {}, number "
                                 "trajectories = {}, nPixPerThread = {})",
                                 p.sequenceLength,
                                 p.vMin,
                                 p.vMax,
                                 p.nTrajectories,
                                 nPixPerThread)});
            }
        }

        // Compare
        for (auto mx1 = std::cbegin(results); mx1 != --std::cend(results); mx1++) {
            for (auto mx2 = mx1 + 1; mx2 != std::cend(results); mx2++) {
                auto const cmp = compareDiffMx(mx1->first, mx2->first);
                INFO(fmt::format("{} -- {} vs {}\n", cmp, mx1->second, mx2->second));

                REQUIRE(cmp.max == Approx{0.0f}.margin(1e-4));
                REQUIRE(cmp.mean == Approx{0.0f}.margin(1e-5));
                REQUIRE(cmp.std == Approx{0.0f}.margin(1e-5));
            }
        }
    }
}
