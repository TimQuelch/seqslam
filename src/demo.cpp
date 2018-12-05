#include "seqslam.h"

#include <filesystem>
#include <iostream>

#include <fmt/format.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>

using namespace seqslam;

int main() {
    auto const dataDir = std::filesystem::path{"../datasets/nordland_trimmed_resized"};
    auto const referenceImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    auto const queryImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "winter"), 20));

    // auto const diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    auto diffMatrix = opencl::generateDiffMx(referenceImages, queryImages, 4);

    diffMatrix *= 255 / diffMatrix.maxCoeff();

    auto const diffMatrixIm = [&]() -> cv::Mat {
        auto im = cv::Mat(diffMatrix.rows(), diffMatrix.cols(), CV_8UC1);
        cv::eigen2cv(diffMatrix, im);
        return im;
    }();

    cv::namedWindow("Difference Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Difference Matrix", diffMatrixIm);
    cv::imwrite("diffmx.jpg", diffMatrixIm);

    while (cv::waitKey(0) != 'q')
        ;
}
