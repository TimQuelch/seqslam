#ifndef SEQSLAM_SEQSLAM_H
#define SEQSLAM_SEQSLAM_H

#include "clutils.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include <Eigen/Dense>

#include <opencv2/core.hpp>

namespace seqslam {
    using PixType = float;
    using Mx = Eigen::Matrix<PixType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vx = Eigen::Matrix<PixType, Eigen::Dynamic, 1>;

    [[nodiscard]] auto readImages(std::filesystem::path const& dir) noexcept
        -> std::vector<cv::Mat>;

    [[nodiscard]] auto contrastEnhancement(std::vector<cv::Mat> const& images,
                                           double threshold) noexcept -> std::vector<cv::Mat>;

    [[nodiscard]] auto convertToEigen(std::vector<cv::Mat> const& images) noexcept
        -> std::vector<Mx>;
    [[nodiscard]] auto convertToCv(std::vector<Mx> const& mxs) noexcept -> std::vector<cv::Mat>;
    [[nodiscard]] auto convertToBuffer(std::vector<Mx> const& mxs) noexcept
        -> std::unique_ptr<PixType[]>;

    namespace cpu {
        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize = 32) noexcept -> Mx;

        [[nodiscard]] auto enhanceDiffMx(Mx const& diffMx, unsigned windowSize) noexcept -> Mx;

        [[nodiscard]] auto sequenceSearch(Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned trajectorySteps) noexcept -> Mx;
    } // namespace cpu

    namespace opencl {
        using namespace std::string_view_literals;
        constexpr auto kernelNames = std::array{"diffMxNDiffs"sv,
                                                "diffMxUnrolledWarpReduce"sv,
                                                "diffMxTwoDiffs"sv,
                                                "diffMxContinuousIndex"sv,
                                                "diffMxParallelSave"sv,
                                                "diffMxNaive"sv};
        constexpr auto diffMatrixPath = "kernels/diff-mx.cl"sv;

        auto createDiffMxContext() -> clutils::Context;

        struct diffMxBuffers {
            clutils::Buffer reference;
            clutils::Buffer query;
            clutils::Buffer diffMx;
        };
        [[nodiscard]] auto createBuffers(clutils::Context& context,
                                         std::size_t nReference,
                                         std::size_t nQuery,
                                         std::size_t nPix) -> diffMxBuffers;

        [[nodiscard]] auto fitsInLocalMemory(std::size_t nPix,
                                             std::size_t tileSize,
                                             std::size_t nPerThread) noexcept -> bool;

        [[nodiscard]] auto isValidParameters(std::size_t nImages,
                                             std::size_t nPix,
                                             std::size_t tileSize,
                                             std::size_t nPixPerThread) noexcept -> bool;

        void writeArgs(clutils::Context& context,
                       diffMxBuffers const& buffers,
                       std::vector<Mx> const& referenceMxs,
                       std::vector<Mx> const& queryMxs,
                       std::size_t tileSize,
                       std::size_t nPerThread,
                       std::string_view kernelName = kernelNames[0]);

        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = kernelNames[0]) -> Mx;

        [[nodiscard]] auto generateDiffMx(clutils::Context& context,
                                          std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = kernelNames[0]) -> Mx;

        [[nodiscard]] auto generateDiffMx(clutils::Context const& context,
                                          clutils::Buffer const& outBuffer,
                                          std::size_t referenceSize,
                                          std::size_t querySize,
                                          std::size_t nPix,
                                          std::size_t tileSize = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = kernelNames[0]) -> Mx;
    } // namespace opencl
} // namespace seqslam

#endif
