#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

namespace seqslam {
    using imgMx = Eigen::Matrix<double, 32, 64>;
    using imgMxVector = std::vector<imgMx, Eigen::aligned_allocator<imgMx>>;
    using diffMx = Eigen::MatrixXd;

    auto readImages(std::filesystem::path dir) -> std::vector<cv::Mat>;

    auto contrastEnhancement(const std::vector<cv::Mat>& images, double threshold)
        -> std::vector<cv::Mat>;

    auto convertToEigen(const std::vector<cv::Mat>& images) -> imgMxVector;
    auto convertToCv(const imgMxVector& images) -> std::vector<cv::Mat>;

    namespace cpu {
        auto generateDiffMx(const imgMxVector& referenceMats, const imgMxVector& queryMats)
            -> diffMx;

        auto enhanceDiffMxContrast(const diffMx& mx) -> diffMx;
    } // namespace cpu
} // namespace seqslam
