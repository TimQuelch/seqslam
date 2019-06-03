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
    auto p = readParametersConfig();

    auto const referenceImages = loadImages(
        p.datasetRoot / p.referencePath, p.imageRows, p.imageCols, p.imageContrastThreshold);
    auto const [queryImages, groundTruth] = dropFrames(
        loadImages(p.datasetRoot / p.queryPath, p.imageRows, p.imageCols, p.imageContrastThreshold),
        {1.0, 1.5},
        50);

    p.nQuery = queryImages.size();
    p.nReference = referenceImages.size();

    auto const minTime = std::chrono::milliseconds{5000};
    auto const prPoints = 50;
    auto const slRange = std::tuple{2, 50, 2};
    auto const wsRange = std::tuple{2, 50, 2};
    auto const ntRange = std::tuple{2, 50, 2};
    auto const slDenseRange = std::tuple{2, 50, 1};
    auto const wsDenseRange = std::tuple{2, 50, 1};
    auto const ntDenseRange = std::tuple{2, 50, 1};

    auto const resultSl = parameterSweep(referenceImages,
                                         queryImages,
                                         p,
                                         groundTruth,
                                         prPoints,
                                         minTime,
                                         slDenseRange,
                                         sequenceLength_t{});

    auto const resultWs = parameterSweep(referenceImages,
                                         queryImages,
                                         p,
                                         groundTruth,
                                         prPoints,
                                         minTime,
                                         wsDenseRange,
                                         patchWindowSize_t{});

    auto const resultNt = parameterSweep(
        referenceImages, queryImages, p, groundTruth, prPoints, minTime, ntDenseRange, nTraj_t{});

    auto const resultSlWs = parameterSweep2d(referenceImages,
                                             queryImages,
                                             p,
                                             groundTruth,
                                             prPoints,
                                             minTime,
                                             slRange,
                                             sequenceLength_t{},
                                             wsRange,
                                             patchWindowSize_t{});

    auto const resultSlNt = parameterSweep2d(referenceImages,
                                             queryImages,
                                             p,
                                             groundTruth,
                                             prPoints,
                                             minTime,
                                             slRange,
                                             sequenceLength_t{},
                                             ntRange,
                                             nTraj_t{});

    auto const resultWsNt = parameterSweep2d(referenceImages,
                                             queryImages,
                                             p,
                                             groundTruth,
                                             prPoints,
                                             minTime,
                                             wsRange,
                                             patchWindowSize_t{},
                                             ntRange,
                                             nTraj_t{});

    writeResultsToFile(resultSl, "sweep-sl.json");
    writeResultsToFile(resultWs, "sweep-ws.json");
    writeResultsToFile(resultNt, "sweep-nt.json");
    writeResultsToFile(resultSlWs, "sweep-sl-ws.json");
    writeResultsToFile(resultSlNt, "sweep-sl-nt.json");
    writeResultsToFile(resultWsNt, "sweep-ws-nt.json");
}
