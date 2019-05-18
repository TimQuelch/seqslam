#include "seqslam.h"
#include "measure.h"

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
    auto const pFixed = p;

    auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 1, 24, 8);
    // auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 4);
    // auto enhanced = cpu::enhanceDiffMx(diffMatrix, p.patchWindowSize);
    // auto sequences = cpu::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);

    cv::imwrite("diffmx.jpg", mxToIm(diffMatrix));
    // cv::imwrite("enhanced.jpg", mxToIm(enhanced));
    // cv::imwrite("sequence.jpg", mxToIm(sequences));

    constexpr auto const prPoints = 100;

    auto const vals = std::vector{5, 10, 15, 20, 25, 30};

    p = pFixed;
    for (auto window : vals) {
        p.patchWindowSize = window;
        auto const enhanced = opencl::enhanceDiffMx(diffMatrix, p.patchWindowSize);
        auto const sequences =
            opencl::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
        auto const pr = prCurve(sequences, groundTruth, prPoints);
        writePrCurveToJson(p, pr, fmt::format("pr-window-{}.json", window));
    }

    p = pFixed;
    for (auto seqLength : vals) {
        p.sequenceLength = seqLength;
        auto const enhanced = opencl::enhanceDiffMx(diffMatrix, p.patchWindowSize);
        auto const sequences =
            opencl::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
        auto const pr = prCurve(sequences, groundTruth, prPoints);
        writePrCurveToJson(p, pr, fmt::format("pr-seqlength-{}.json", seqLength));
    }

    p = pFixed;
    for (auto nTraj : vals) {
        p.nTraj = nTraj;
        auto const enhanced = opencl::enhanceDiffMx(diffMatrix, p.patchWindowSize);
        auto const sequences =
            opencl::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
        auto const pr = prCurve(sequences, groundTruth, prPoints);
        writePrCurveToJson(p, pr, fmt::format("pr-ntraj-{}.json", nTraj));
    }
}
