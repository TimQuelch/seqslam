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

const auto smallImagesDir = std::filesystem::path{"datasets/nordland_trimmed_resized"};
const auto largeImagesDir = std::filesystem::path{"datasets/nordland_trimmed"};

struct DiffMxComparison {
    float max;
    float mean;
    float std;
};
auto compareDiffMx(Mx const& one, Mx const& two) {
    Mx const difference = (one - two).cwiseAbs();
    auto const max = difference.maxCoeff();
    auto const mean = difference.mean();
    auto const std =
        std::sqrt((difference.array() - mean).sum() / (difference.rows() * difference.cols() - 1));
    return DiffMxComparison{max, mean, std};
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

TEST_CASE("Difference matrix results consistent with all parameters", "[diffmx]") {
    for (auto shortSide = 1; shortSide <= 32; shortSide++) {
        auto const nPix = 2 * shortSide * shortSide;
        auto const nWarps = nPix / 32;
        auto const size = std::pair(shortSide, 2 * shortSide);

        if (nPix % 32 != 0 || (nWarps & (nWarps - 1))) {
            continue;
        }

        auto const referenceImages = convertToEigen(
            contrastEnhancement(resizeImages(readImages(largeImagesDir / "summer"), size), 20));
        auto const queryImages = convertToEigen(
            contrastEnhancement(resizeImages(readImages(largeImagesDir / "winter"), size), 20));

        assert(!referenceImages.empty());
        assert(!queryImages.empty());
        assert(referenceImages.size() == queryImages.size());
        assert(referenceImages[0].rows() == queryImages[0].rows());
        assert(referenceImages[0].cols() == queryImages[0].cols());
        auto const nImages = referenceImages.size();

        auto diffMxFns = std::vector<std::pair<std::function<Mx()>, std::string>>{};
        for (auto tileSize = 1u; tileSize <= 256; tileSize *= 2) {
            diffMxFns.push_back(
                {[&referenceImages, &queryImages, tileSize]() {
                     return cpu::generateDiffMx(referenceImages, queryImages, tileSize);
                 },
                 fmt::format(
                     "{}x{} ({}) CPU tile size {}", shortSide, 2 * shortSide, nPix, tileSize)});
        }
        for (auto tileSize = 1u; tileSize <= 512; tileSize++) {
            for (auto nPerThread = 1u; nPerThread <= 512; nPerThread++) {
                if (opencl::diffmxcalc::isValidParameters(nImages, nPix, tileSize, nPerThread)) {
                    diffMxFns.push_back(
                        {[&referenceImages, &queryImages, tileSize, nPerThread]() {
                             return opencl::generateDiffMx(
                                 referenceImages, queryImages, tileSize, nPerThread);
                         },
                         fmt::format("{}x{} ({}) GPU tile size {} n pixels per thread {}",
                                     shortSide,
                                     2 * shortSide,
                                     nPix,
                                     tileSize,
                                     nPerThread)});
                }
            }
        }

        auto diffMxs = std::vector<std::pair<Mx, std::string>>{};
        std::transform(std::cbegin(diffMxFns),
                       std::cend(diffMxFns),
                       std::back_inserter(diffMxs),
                       [](auto const& pair) {
                           return std::pair{pair.first(), pair.second};
                       });

        for (auto mx1 = std::cbegin(diffMxs); mx1 != --std::cend(diffMxs); mx1++) {
            for (auto mx2 = mx1 + 1; mx2 != std::cend(diffMxs); mx2++) {
                auto const cmp = compareDiffMx(mx1->first, mx2->first);
                INFO(fmt::format("{} -- {} vs {}\n", cmp, mx1->second, mx2->second));

                REQUIRE(cmp.max == Approx{0.0});
                REQUIRE(cmp.mean == Approx{0.0});
                REQUIRE(cmp.std == Approx{0.0});
            }
        }
    }
}
