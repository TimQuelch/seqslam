#include "seqslam.h"

#include <filesystem>
#include <functional>
#include <memory>

using namespace seqslam;
using namespace std::literals::string_view_literals;

const auto smallImagesDir = std::filesystem::path{"../datasets/nordland_trimmed_resized"};
const auto largeImagesDir = std::filesystem::path{"../datasets/nordland_trimmed"};

int main() {
    auto const referenceImages =
        convertToEigen(contrastEnhancement(readImages(largeImagesDir / "summer"), 20));
    auto const queryImages =
        convertToEigen(contrastEnhancement(readImages(largeImagesDir / "winter"), 20));

    auto diffMxFns = std::vector<std::pair<std::function<std::unique_ptr<Mx>()>, std::string>>{};
    for (auto i = 1u; i <= 8; i = i * 2) {
        diffMxFns.push_back({[&referenceImages, &queryImages, i]() {
                                 return cpu::generateDiffMx(referenceImages, queryImages, i);
                             },
                             fmt::format("CPU tile size {}", i)});
    }

    diffMxFns.push_back({[&referenceImages, &queryImages]() {
                             return opencl::generateDiffMx(referenceImages, queryImages, 1);
                         },
                         fmt::format("GPU tile size {}", 1)});

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
