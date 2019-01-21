#include "seqslam.h"

#include <cassert>
#include <filesystem>
#include <functional>
#include <memory>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <catch2/catch.hpp>

using namespace seqslam;
using namespace std::literals::string_view_literals;

constexpr auto const smallImagesDir = "datasets/nordland_trimmed_resized"sv;
constexpr auto const largeImagesDir = "datasets/nordland_trimmed"sv;

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
        return cpu::generateDiffMx(std::get<0>(imgs), std::get<1>(imgs));
    }();
    return diffMx;
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

TEST_CASE("Difference matrix enhancement results consistent with all parameters",
          "[diffmxenhancement]") {
    auto const mx = getDiffMx();

    for (auto windowSize = 4u; windowSize <= 256u;
         windowSize = windowSize < 64u ? windowSize * 2u : windowSize + 64u) {

        auto results = std::vector<std::pair<Mx, std::string>>{};

        // Compute with CPU
        results.push_back(
            {cpu::enhanceDiffMx(mx, windowSize), fmt::format("CPU windowSize {}", windowSize)});

        // Compute with GPU
        for (auto nPixPerThread = 1u; nPixPerThread <= 50u; nPixPerThread++) {
            if (opencl::diffmxenhance::isValidParameters(mx.rows(), nPixPerThread)) {
                results.push_back(
                    {opencl::enhanceDiffMx(mx, windowSize, nPixPerThread),
                     fmt::format("GPU windowSize {} nPixPerThread {}", windowSize, nPixPerThread)});
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
