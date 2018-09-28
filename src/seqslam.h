#ifndef SEQSLAM_SEQSLAM_H
#define SEQSLAM_SEQSLAM_H

#include "clutils.h"

#include <filesystem>
#include <memory>
#include <vector>

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

    auto contrastEnhancement(std::vector<cv::Mat> const& images, double threshold)
        -> std::vector<cv::Mat>;

    auto convertToEigen(std::vector<cv::Mat> const& images) -> ImgMxVector;
    auto convertToCv(ImgMxVector const& images) -> std::vector<cv::Mat>;
    auto convertToBuffer(ImgMxVector const& images) -> std::unique_ptr<PixType[]>;

    namespace cpu {
        auto generateDiffMx(ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize = 32) -> std::unique_ptr<DiffMx>;
    } // namespace cpu

    namespace opencl {
        auto const diffMatrixKernel = std::string{"diffMx"};
        auto const diffMatrixPath = std::filesystem::path{"kernels/diff-mx.cl"};

        auto createDiffMxContext() -> clutils::Context;

        struct diffMxBuffers {
            clutils::Buffer reference;
            clutils::Buffer query;
            clutils::Buffer diffMx;
        };
        auto createBuffers(clutils::Context& context, std::size_t nReference, std::size_t nQuery)
            -> diffMxBuffers;

        void writeArgs(clutils::Context& context,
                       diffMxBuffers const& buffers,
                       ImgMxVector const& referenceMxs,
                       ImgMxVector const& queryMxs,
                       std::size_t tileSize);

        auto generateDiffMx(ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx>;

        auto generateDiffMx(clutils::Context& context,
                            ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx>;

        auto generateDiffMx(clutils::Context const& context,
                            clutils::Buffer const& outBuffer,
                            std::size_t referenceSize,
                            std::size_t querySize,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx>;
    } // namespace opencl
} // namespace seqslam

#endif
