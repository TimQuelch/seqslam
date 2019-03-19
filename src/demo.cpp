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

    auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    // auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 4);
    auto enhanced = cpu::enhanceDiffMx(diffMatrix, 10);
    auto sequences = cpu::sequenceSearch(enhanced, 15, 0.8, 1.2, 15);

    cv::imwrite("diffmx.jpg", mxToIm(diffMatrix));
    cv::imwrite("enhanced.jpg", mxToIm(enhanced));
    cv::imwrite("sequence.jpg", mxToIm(sequences));

    auto const pr = prCurve(sequences, nordlandGroundTruth(referenceImages.size()), 30);
    writePrCurveToCsv(pr, "pr.csv");
    writePrCurveToJson(pr, "pr.json");
}
