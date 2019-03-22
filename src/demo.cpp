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

int main() {
    auto const dataDir = std::filesystem::path{"datasets/nordland_trimmed_resized"};
    auto const referenceImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto const queryImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

    auto p = seqslamParameters{};
    p.nPix = referenceImages[0].rows() * referenceImages[0].cols();
    p.nQuery = queryImages.size();
    p.nReference = referenceImages.size();
    p.patchWindowSize = 10;
    p.sequenceLength = 15;
    p.vMin = 0.8;
    p.vMax = 1.2;
    p.nTraj = 15;

    auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    // auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 4);
    auto enhanced = cpu::enhanceDiffMx(diffMatrix, p.patchWindowSize);
    auto sequences = cpu::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);

    cv::imwrite("diffmx.jpg", mxToIm(diffMatrix));
    cv::imwrite("enhanced.jpg", mxToIm(enhanced));
    cv::imwrite("sequence.jpg", mxToIm(sequences));

    auto const pr = prCurve(sequences, nordlandGroundTruth(referenceImages.size()), 30);
    writePrCurveToJson(p, pr, "pr.json");
}
