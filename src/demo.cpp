#include "measure.h"
#include "seqslam.h"

#include <filesystem>
#include <iostream>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>

using namespace seqslam;

cv::Mat mxToIm(Mx const& mx) {
    Mx scaled = mx.array() - mx.minCoeff();
    scaled *= 255 / scaled.maxCoeff();
    auto im = cv::Mat(scaled.rows(), scaled.cols(), CV_8UC1);
    cv::eigen2cv(scaled, im);
    return im;
}

[[nodiscard]] auto loadImages(std::filesystem::path const& path,
                              unsigned rows,
                              unsigned cols,
                              unsigned contrastThreshold) {
    auto const images = readImages(path);
    auto const resized = resizeImages(images, {rows, cols});
    auto const enhanced = contrastEnhancement(resized, contrastThreshold);
    return convertToEigen(enhanced);
}

int main() {
    auto p = readParametersConfig(utils::defaultConfig);

    auto const referenceImages = loadImages(
        p.datasetRoot / p.referencePath, p.imageRows, p.imageCols, p.imageContrastThreshold);
    auto const [queryImages, groundTruth] = dropFrames(
        loadImages(p.datasetRoot / p.queryPath, p.imageRows, p.imageCols, p.imageContrastThreshold),
        {1.0, 1.5},
        30);

    p.nQuery = queryImages.size();
    p.nReference = referenceImages.size();

    constexpr auto const prPoints = 5;
    constexpr auto const minTime = std::chrono::milliseconds{200};
    constexpr auto const range = std::pair{9, 11};

    auto const result = parameterSweep(
        referenceImages, queryImages, p, groundTruth, prPoints, minTime, range, sequenceLength_t{});
    writeResultsToFile(result, "sweep.json");

    auto const result2 = parameterSweep2d(referenceImages,
                                          queryImages,
                                          p,
                                          groundTruth,
                                          prPoints,
                                          minTime,
                                          range,
                                          sequenceLength_t{},
                                          range,
                                          nTraj_t{});
    writeResultsToFile(result2, "sweep2.json");
}
