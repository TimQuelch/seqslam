#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

namespace seqslam {
    constexpr auto nRows = 32u;
    constexpr auto nCols = 64u;

    using ImgMx = Eigen::Matrix<float, nRows, nCols>;
    using ImgMxVector = std::vector<ImgMx, Eigen::aligned_allocator<ImgMx>>;
    using DiffMx = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

    auto readImages(std::filesystem::path dir) -> std::vector<cv::Mat>;

    auto contrastEnhancement(const std::vector<cv::Mat>& images, double threshold)
        -> std::vector<cv::Mat>;

    auto convertToEigen(const std::vector<cv::Mat>& images) -> ImgMxVector;
    auto convertToCv(const ImgMxVector& images) -> std::vector<cv::Mat>;

    namespace cpu {
        auto generateDiffMx(const ImgMxVector& referenceMxs,
                            const ImgMxVector& queryMxs,
                            std::size_t tileSize = 32) -> std::unique_ptr<DiffMx>;
    } // namespace cpu
} // namespace seqslam
