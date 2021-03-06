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
    using Mx = Eigen::Matrix<PixType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using MxBool = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Vx = Eigen::Matrix<PixType, Eigen::Dynamic, 1>;

    [[nodiscard]] auto readImages(std::filesystem::path const& dir) noexcept
        -> std::vector<cv::Mat>;

    [[nodiscard]] auto resizeImages(std::vector<cv::Mat> const& images, std::pair<int, int> size)
        -> std::vector<cv::Mat>;

    [[nodiscard]] auto contrastEnhancement(std::vector<cv::Mat> const& images,
                                           double threshold) noexcept -> std::vector<cv::Mat>;

    [[nodiscard]] auto convertToEigen(std::vector<cv::Mat> const& images) noexcept
        -> std::vector<Mx>;
    [[nodiscard]] auto convertToCv(std::vector<Mx> const& mxs) noexcept -> std::vector<cv::Mat>;
    [[nodiscard]] auto convertToBuffer(std::vector<Mx> const& mxs) noexcept
        -> std::unique_ptr<PixType[]>;

    [[nodiscard]] auto predict(Mx const& mx, float threshold) -> std::vector<std::vector<unsigned>>;

    namespace cpu {
        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSizeR = 32,
                                          std::size_t tileSizeQ = 32) noexcept -> Mx;

        [[nodiscard]] auto enhanceDiffMx(Mx const& diffMx, unsigned windowSize) noexcept -> Mx;

        [[nodiscard]] auto sequenceSearch(Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned trajectorySteps) noexcept -> Mx;
    } // namespace cpu

    namespace opencl {
        using namespace std::string_view_literals;

        constexpr auto localMemorySize = 48u * 1024u;

        [[nodiscard]] auto createContext() -> clutils::Context;

        namespace diffmxcalc {
            constexpr auto kernels = std::pair{"kernels/diff-mx.cl"sv,
                                               std::array{"diffMxNDiffs"sv,
                                                          "diffMxUnrolledWarpReduce"sv,
                                                          "diffMxTwoDiffs"sv,
                                                          "diffMxContinuousIndex"sv,
                                                          "diffMxParallelSave"sv,
                                                          "diffMxNaive"sv}};
            constexpr auto defaultKernel = kernels.second[0];

            [[nodiscard]] auto createContext() -> clutils::Context;

            struct diffMxBuffers {
                clutils::Buffer reference;
                clutils::Buffer query;
                clutils::Buffer diffMx;
            };
            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             std::size_t nReference,
                                             std::size_t nQuery,
                                             std::size_t nPix) -> diffMxBuffers;

            [[nodiscard]] auto isValidParameters(std::size_t nReference,
                                                 std::size_t nQuery,
                                                 std::size_t nPix,
                                                 std::size_t tileSizeR,
                                                 std::size_t tileSizeQ,
                                                 std::size_t nPixPerThread) noexcept -> bool;

            void writeArgs(clutils::Context& context,
                           diffMxBuffers const& buffers,
                           std::vector<Mx> const& referenceMxs,
                           std::vector<Mx> const& queryMxs,
                           std::size_t tileSizeR,
                           std::size_t tileSizeQ,
                           std::size_t nPerThread,
                           std::string_view kernelName = defaultKernel);
        } // namespace diffmxcalc

        namespace diffmxenhance {
            constexpr auto kernels =
                std::pair{"kernels/enhancement.cl"sv,
                          std::array{"enhanceDiffMxSequential"sv, "enhanceDiffMxStrided"sv}};
            constexpr auto defaultKernel = kernels.second[0];

            [[nodiscard]] auto createContext() -> clutils::Context;

            struct diffMxEnhanceBuffers {
                clutils::Buffer in;
                clutils::Buffer out;
            };
            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             std::size_t nReference,
                                             std::size_t nQuery) -> diffMxEnhanceBuffers;

            [[nodiscard]] auto isValidParameters(std::size_t nReference,
                                                 unsigned nPixPerThread) noexcept -> bool;

            void writeArgs(clutils::Context& context,
                           diffMxEnhanceBuffers const& buffers,
                           Mx const& diffMx,
                           int windowSize,
                           unsigned nPixPerThread,
                           std::string_view kernelName = defaultKernel);
        } // namespace diffmxenhance

        namespace seqsearch {
            constexpr auto kernels =
                std::pair{"kernels/search.cl"sv, std::array{"sequenceSearch"sv}};
            constexpr auto defaultKernel = kernels.second[0];

            [[nodiscard]] auto createContext() -> clutils::Context;

            struct sequenceSearchBuffers {
                clutils::Buffer in;
                clutils::Buffer qOffsets;
                clutils::Buffer rOffsets;
                clutils::Buffer out;
            };
            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             unsigned sequenceLength,
                                             unsigned nTrajectories,
                                             std::size_t nReference,
                                             std::size_t nQuery) -> sequenceSearchBuffers;

            [[nodiscard]] auto isValidParameters(std::size_t nReference,
                                                 unsigned nPixPerThread) noexcept -> bool;

            void writeArgs(clutils::Context& context,
                           sequenceSearchBuffers const& buffers,
                           Mx const& diffMx,
                           unsigned sequenceLength,
                           float vMin,
                           float vMax,
                           unsigned nTrajectories,
                           unsigned nPixPerThread,
                           std::string_view kernelName = defaultKernel);
        } // namespace seqsearch

        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSizeR = 4,
                                          std::size_t tileSizeQ = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = diffmxcalc::defaultKernel)
            -> Mx;

        [[nodiscard]] auto generateDiffMx(clutils::Context& context,
                                          std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSizeR = 4,
                                          std::size_t tileSizeQ = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = diffmxcalc::defaultKernel)
            -> Mx;

        [[nodiscard]] auto generateDiffMx(clutils::Context const& context,
                                          clutils::Buffer const& outBuffer,
                                          std::size_t referenceSize,
                                          std::size_t querySize,
                                          std::size_t nPix,
                                          std::size_t tileSizeR = 4,
                                          std::size_t tileSizeQ = 4,
                                          std::size_t nPerThread = 4,
                                          std::string_view kernelName = diffmxcalc::defaultKernel)
            -> Mx;

        [[nodiscard]] auto enhanceDiffMx(Mx const& diffMx,
                                         unsigned windowSize,
                                         unsigned nPixPerThread = 4,
                                         std::string_view kernelName = diffmxenhance::defaultKernel)
            -> Mx;

        [[nodiscard]] auto enhanceDiffMx(clutils::Context& context,
                                         Mx const& diffMx,
                                         unsigned windowSize,
                                         unsigned nPixPerThread = 4,
                                         std::string_view kernelName = diffmxenhance::defaultKernel)
            -> Mx;

        [[nodiscard]] auto enhanceDiffMx(clutils::Context const& context,
                                         clutils::Buffer const& outBuffer,
                                         std::size_t nReference,
                                         std::size_t nQuery,
                                         unsigned nPixPerThread = 4,
                                         std::string_view kernelName = diffmxenhance::defaultKernel)
            -> Mx;

        [[nodiscard]] auto sequenceSearch(Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned nTrajectories,
                                          unsigned nPixPerThread = 4,
                                          std::string_view kernelName = seqsearch::defaultKernel)
            -> Mx;

        [[nodiscard]] auto sequenceSearch(clutils::Context context,
                                          Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned nTrajectories,
                                          unsigned nPixPerThread = 4,
                                          std::string_view kernelName = seqsearch::defaultKernel)
            -> Mx;

        [[nodiscard]] auto sequenceSearch(clutils::Context const& context,
                                          clutils::Buffer const& outBuffer,
                                          std::size_t nReference,
                                          std::size_t nQuery,
                                          unsigned nPixPerThread = 4,
                                          std::string_view kernelName = seqsearch::defaultKernel)
            -> Mx;
    } // namespace opencl
} // namespace seqslam

#endif
