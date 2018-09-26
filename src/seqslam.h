#ifndef SEQSLAM_SEQSLAM_H
#define SEQSLAM_SEQSLAM_H

#include "clutils.h"

#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

namespace seqslam {
    constexpr auto nRows = 16u;
    constexpr auto nCols = 32u;

    using PixType = float;
    using ImgMx = Eigen::Matrix<PixType, nRows, nCols, Eigen::RowMajor>;
    using ImgMxVector = std::vector<ImgMx, Eigen::aligned_allocator<ImgMx>>;
    using DiffMx = Eigen::Matrix<PixType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

    namespace opencl {
        auto convertToBuffer(const ImgMxVector& images, Context& context, Buffer::Access access)
            -> Buffer;
    } // namespace opencl
} // namespace seqslam

#endif
