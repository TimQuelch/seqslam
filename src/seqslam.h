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

    auto readImages(std::filesystem::path const& dir) -> std::vector<cv::Mat>;

    auto contrastEnhancement(std::vector<cv::Mat> const& images, double threshold)
        -> std::vector<cv::Mat>;

    auto convertToEigen(std::vector<cv::Mat> const& images) -> std::vector<Mx>;
    auto convertToCv(std::vector<Mx> const& mxs) -> std::vector<cv::Mat>;
    auto convertToBuffer(std::vector<Mx> const& mxs) -> std::unique_ptr<PixType[]>;

    struct DiffMxComparison {
        float max;
        float mean;
        float std;
    };
    auto compareDiffMx(Mx const& one, Mx const& two) -> DiffMxComparison;

    namespace cpu {
        auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize = 32) -> std::unique_ptr<Mx>;
    } // namespace cpu

    namespace opencl {
        using namespace std::literals::string_literals;
        auto const kernelNames =
            std::vector{"diffMx"s, "diffMxStridedIndex"s, "diffMxSerialSave"s, "diffMxNoUnroll"s};
        auto const diffMatrixPath = std::filesystem::path{"kernels/diff-mx.cl"};

        auto createDiffMxContext() -> clutils::Context;

        struct diffMxBuffers {
            clutils::Buffer reference;
            clutils::Buffer query;
            clutils::Buffer diffMx;
        };
        auto createBuffers(clutils::Context& context,
                           std::size_t nReference,
                           std::size_t nQuery,
                           std::size_t nPix) -> diffMxBuffers;

        auto fitsInLocalMemory(std::size_t nPix, std::size_t tileSize, std::size_t nPerThread)
            -> std::tuple<std::size_t, bool>;

        void writeArgs(clutils::Context& context,
                       diffMxBuffers const& buffers,
                       std::vector<Mx> const& referenceMxs,
                       std::vector<Mx> const& queryMxs,
                       std::size_t tileSize,
                       std::size_t nPerThread,
                       std::string const& kernelName = kernelNames[0]);

        auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize = 4,
                            std::size_t nPerThread = 4,
                            std::string const& kernelName = kernelNames[0]) -> std::unique_ptr<Mx>;

        auto generateDiffMx(clutils::Context& context,
                            std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize = 4,
                            std::size_t nPerThread = 4,
                            std::string const& kernelName = kernelNames[0]) -> std::unique_ptr<Mx>;

        auto generateDiffMx(clutils::Context const& context,
                            clutils::Buffer const& outBuffer,
                            std::size_t referenceSize,
                            std::size_t querySize,
                            std::size_t nPix,
                            std::size_t tileSize = 4,
                            std::size_t nPerThread = 4,
                            std::string const& kernelName = kernelNames[0]) -> std::unique_ptr<Mx>;
    } // namespace opencl
} // namespace seqslam

namespace fmt {
    template <>
    struct formatter<seqslam::DiffMxComparison> {
        template <typename ParseContext>
        constexpr auto parse(ParseContext& context) {
            return context.begin();
        }

        template <typename FormatContext>
        auto format(seqslam::DiffMxComparison const& c, FormatContext& context) {
            return format_to(
                context.begin(), "Max diff: {}, Mean diff {}, Std diff {}", c.max, c.mean, c.std);
        }
    };
} // namespace fmt

#endif
