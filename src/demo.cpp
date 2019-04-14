#include "seqslam.h"
#include "utils.h"

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

[[nodiscard]] auto readParametersConfig(std::filesystem::path const& configFile) {
    auto const json =
        utils::readJsonConfig(utils::traverseUpUntilMatch(configFile).value())["seqslam"];
    auto p = seqslamParameters{};
    p.datasetRoot = json["Dataset root path"].get<std::string>();
    p.queryPath = json["Query image path"].get<std::string>();
    p.referencePath = json["Reference image path"].get<std::string>();
    p.imageRows = json["Image rows"];
    p.imageCols = json["Image columns"];
    p.imageContrastThreshold = json["Image contrast enhancement threshold"];
    p.patchWindowSize = json["Patch window size"];
    p.sequenceLength = json["Sequence length"];
    p.vMin = json["Min velocity"];
    p.vMax = json["Max velocity"];
    p.nTraj = json["Number of trajectories"];

    return p;
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

    auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    // auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 4);
    // auto enhanced = cpu::enhanceDiffMx(diffMatrix, p.patchWindowSize);
    // auto sequences = cpu::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);

    // cv::imwrite("diffmx.jpg", mxToIm(diffMatrix));
    // cv::imwrite("enhanced.jpg", mxToIm(enhanced));
    // cv::imwrite("sequence.jpg", mxToIm(sequences));

    for (auto window : {5, 10, 15, 20, 30}) {
        p.patchWindowSize = window;
        auto const enhanced = cpu::enhanceDiffMx(diffMatrix, p.patchWindowSize);
        auto const sequences =
            cpu::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
        auto const pr = prCurve(sequences, nordlandGroundTruth(referenceImages.size()), 30);
        writePrCurveToJson(p, pr, fmt::format("pr-{}.json", window));
    }
}
