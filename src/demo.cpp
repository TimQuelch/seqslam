#include "seqslam.h"

#include <filesystem>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>

using namespace seqslam;

int main() {
    const auto dataDir = std::filesystem::path{"../datasets/nordland_trimmed"};
    const auto referenceImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));
    const auto queryImages =
        convertToEigen(contrastEnhancement(readImages(dataDir / "summer"), 20));

    const auto diffMatrix = cpu::generateDiffMx(referenceImages, queryImages);
    const auto diffMatrixEnhanced = cpu::enhanceDiffMxContrast(diffMatrix);

    const auto diffMatrixIm = [&]() -> cv::Mat {
        auto im = cv::Mat(diffMatrixEnhanced.rows(), diffMatrixEnhanced.cols(), CV_8UC1);
        cv::eigen2cv(diffMx{diffMatrixEnhanced * 255}, im);
        return im;
    }();

    cv::namedWindow("Difference Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Difference Matrix", diffMatrixIm);
    cv::imwrite("diffmx.jpg", diffMatrixIm);

    while (cv::waitKey(0) != 'q')
        ;
}
