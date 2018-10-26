#include "seqslam.h"

#include <cassert>
#include <filesystem>
#include <functional>
#include <memory>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace seqslam;
using namespace std::literals::string_view_literals;

const auto smallImagesDir = std::filesystem::path{"../datasets/nordland_trimmed_resized"};
const auto largeImagesDir = std::filesystem::path{"../datasets/nordland_trimmed"};

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

int main() {
    auto const referenceImages =
        convertToEigen(contrastEnhancement(readImages(largeImagesDir / "summer"), 20));
    auto const queryImages =
        convertToEigen(contrastEnhancement(readImages(largeImagesDir / "winter"), 20));

    assert(referenceImages.size() > 0);
    assert(queryImages.size() > 0);
    assert(referenceImages.size() == queryImages.size());
    assert(referenceImages[0].rows() == queryImages[0].rows());
    assert(referenceImages[0].cols() == queryImages[0].cols());
    auto const nPix = referenceImages[0].rows() * referenceImages[0].cols();
    auto const nImages = referenceImages.size();

    auto diffMxFns = std::vector<std::pair<std::function<std::unique_ptr<Mx>()>, std::string>>{};
    for (auto tileSize = 1u; tileSize <= 8; tileSize *= 2) {
        diffMxFns.push_back({[&referenceImages, &queryImages, tileSize]() {
                                 return cpu::generateDiffMx(referenceImages, queryImages, tileSize);
                             },
                             fmt::format("CPU tile size {}", tileSize)});
    }
    for (auto tileSize = 1u; tileSize <= 50; tileSize++) {
        for (auto nPerThread = 1u; nPerThread <= 50; nPerThread++) {
            bool const fits = opencl::fitsInLocalMemory(nPix, tileSize, nPerThread);
            bool const nThreads = nPix / nPerThread < 1024;
            bool const tilesCorrectly = !(nImages % tileSize);
            bool const initialReduceCorrectly = !(nPix % nPerThread);
            if (fits && nThreads && tilesCorrectly && initialReduceCorrectly) {
                diffMxFns.push_back(
                    {[&referenceImages, &queryImages, tileSize, nPerThread]() {
                         return opencl::generateDiffMx(
                             referenceImages, queryImages, tileSize, nPerThread);
                     },
                     fmt::format("GPU tile size {} n pixels per thread {}", tileSize, nPerThread)});
            }
        }
    }

    auto diffMxs = std::vector<std::pair<std::unique_ptr<Mx>, std::string>>{};
    std::transform(std::cbegin(diffMxFns),
                   std::cend(diffMxFns),
                   std::back_inserter(diffMxs),
                   [](auto const& pair) {
                       return std::pair{pair.first(), pair.second};
                   });

    for (auto mx1 = std::cbegin(diffMxs); mx1 != --std::cend(diffMxs); mx1++) {
        for (auto mx2 = mx1 + 1; mx2 != std::cend(diffMxs); mx2++) {
            fmt::print("{} -- {} vs {}\n",
                       compareDiffMx(*(mx1->first), *(mx2->first)),
                       mx1->second,
                       mx2->second);
        }
    }
}
