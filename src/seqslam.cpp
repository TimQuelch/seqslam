#include "seqslam.h"
#include "seqslam-detail.h"

#include "utils.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <limits>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace seqslam {
    namespace detail {
        struct Dims {
            int rows;
            int cols;
            constexpr auto nElems() const { return rows * cols; }
        };

        [[nodiscard]] auto dims(Mx const& mx) noexcept {
            return Dims{static_cast<int>(mx.rows()), static_cast<int>(mx.cols())};
        }

        [[nodiscard]] constexpr auto dims(cv::Mat const& mx) noexcept {
            return Dims{static_cast<int>(mx.rows), static_cast<int>(mx.cols)};
        }

        [[maybe_unused]] [[nodiscard]] constexpr auto operator==(Dims const& one,
                                                                 Dims const& two) noexcept {
            return one.rows == two.rows && one.cols == two.cols;
        }

        [[nodiscard]] auto calcTrajectoryQueryIndexOffsets(unsigned seqLength) noexcept
            -> std::vector<int> {
            auto qi = std::vector<int>(seqLength);
            std::iota(qi.begin(), qi.end(), -1 * std::floor(static_cast<int>(seqLength) / 2));
            return qi;
        }

        [[nodiscard]] auto calcTrajectoryReferenceIndexOffsets(std::vector<int> const& qi,
                                                               float vMin,
                                                               float vMax,
                                                               unsigned nSteps) noexcept
            -> std::vector<std::vector<int>> {
            auto const seqLength = qi.size();
            auto const uniqueVs = [trajMin = std::round(qi.front() * vMin),
                                   trajMax = std::round(qi.front() * vMax),
                                   seqLength]() {
                auto uniqueStarts = std::vector<int>();
                std::generate_n(std::back_inserter(uniqueStarts),
                                trajMin - trajMax + 1,
                                [n = trajMin]() mutable { return n--; });
                assert(uniqueStarts.front() == trajMin);
                assert(uniqueStarts.back() == trajMax);

                auto uniqueVelocities = std::vector<float>();
                std::transform(uniqueStarts.begin(),
                               uniqueStarts.end(),
                               std::back_inserter(uniqueVelocities),
                               [seqLength](auto val) {
                                   return -1 * val /
                                          std::floor(static_cast<float>(seqLength) / 2.0f);
                               });
                return uniqueVelocities;
            }();

            auto const vs = [&uniqueVs,
                             vMin,
                             vMax,
                             vStep = (vMax - vMin) / static_cast<float>((nSteps - 1))]() {
                auto vels = std::vector<float>{};
                auto diffs = std::vector<float>(uniqueVs.size());

                for (auto v = vMin; v <= vMax; v += vStep) {
                    std::transform(uniqueVs.begin(),
                                   uniqueVs.end(),
                                   diffs.begin(),
                                   [v](auto vecVal) { return std::abs(vecVal - v); });
                    auto const closest =
                        std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
                    if (std::find(vels.begin(), vels.end(), uniqueVs[closest]) == vels.end()) {
                        vels.push_back(uniqueVs[closest]);
                    }
                }
                return vels;
            }();

            auto ri = std::vector<std::vector<int>>();
            auto traj = std::vector<int>(qi.size());
            ri.reserve(vs.size());
            std::transform(vs.begin(), vs.end(), std::back_inserter(ri), [&qi, &traj](auto v) {
                std::transform(
                    qi.begin(), qi.end(), traj.begin(), [v](auto q) { return std::round(q * v); });
                return traj;
            });

            return ri;
        }

        [[nodiscard]] auto calcTrajectoryScore(Mx const& diffMx,
                                               unsigned r,
                                               unsigned q,
                                               std::vector<int> const& rTrajectory,
                                               std::vector<int> const& qTrajectory) noexcept
            -> float {
            assert(qTrajectory.size() == rTrajectory.size());

            auto score = 0.0f;
            for (auto i = 0u; i < qTrajectory.size(); i++) {
                score += diffMx(r + rTrajectory[i], q + qTrajectory[i]);
            }

            return score;
        }

        [[nodiscard]] auto concatenateROffsets(std::vector<std::vector<int>> rOffsets) {
            assert(!rOffsets.empty());
            auto const sequenceLength = rOffsets[0].size();
            std::for_each(rOffsets.begin(),
                          rOffsets.end(),
                          [sequenceLength]([[maybe_unused]] auto const& rOffset) {
                              static_cast<void>(sequenceLength); // Squash warning in release build
                              assert(rOffset.size() == sequenceLength);
                          });

            auto concatenated = std::vector<int>{};
            concatenated.reserve(sequenceLength * rOffsets.size());
            for (auto const& rOffset : rOffsets) {
                std::copy(std::begin(rOffset), std::end(rOffset), std::back_inserter(concatenated));
            }
            assert(concatenated.size() == sequenceLength * rOffsets.size());
            return concatenated;
        }

        [[nodiscard]] auto findOrThrow(std::filesystem::path const& suffix) {
            if (auto path = utils::traverseUpUntilMatch(suffix)) {
                return *path;
            } else {
                throw std::runtime_error{fmt::format("{} not found", suffix.string())};
            }
        }
    } // namespace detail
    using namespace detail;

    [[nodiscard]] auto readImages(std::filesystem::path const& dir) noexcept
        -> std::vector<cv::Mat> {
        std::vector<cv::Mat> images;
        for (auto const& imagePath : std::filesystem::directory_iterator(findOrThrow(dir))) {
            images.push_back(cv::imread(imagePath.path().string(), cv::IMREAD_GRAYSCALE));
        }
        return images;
    }

    [[nodiscard]] auto contrastEnhancement(std::vector<cv::Mat> const& images,
                                           double threshold) noexcept -> std::vector<cv::Mat> {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(threshold);

        std::vector<cv::Mat> contrastEnhanced;
        std::transform(images.begin(),
                       images.end(),
                       std::back_inserter(contrastEnhanced),
                       [&clahe](auto const& im) {
                           cv::Mat res;
                           clahe->apply(im, res);
                           return res;
                       });
        return contrastEnhanced;
    }

    [[nodiscard]] auto convertToEigen(std::vector<cv::Mat> const& images) noexcept
        -> std::vector<Mx> {
        assert(!images.empty());
        auto const d = dims(images[0]);
        std::vector<Mx> res;
        res.reserve(images.size());
        std::transform(images.begin(), images.end(), std::back_inserter(res), [d](auto const& im) {
            assert(im.rows == d.rows);
            assert(im.cols == d.cols);
            Mx mat{d.rows, d.cols};
            cv::cv2eigen(im, mat);
            return mat;
        });
        return res;
    }

    [[nodiscard]] auto convertToCv(std::vector<Mx> const& mxs) noexcept -> std::vector<cv::Mat> {
        std::vector<cv::Mat> res;
        res.reserve(mxs.size());
        std::transform(mxs.begin(), mxs.end(), std::back_inserter(res), [](auto const& mx) {
            cv::Mat im;
            cv::eigen2cv(mx, im);
            return im;
        });
        return res;
    }

    [[nodiscard]] auto convertToBuffer(std::vector<Mx> const& mxs) noexcept
        -> std::unique_ptr<PixType[]> {
        assert(!mxs.empty());
        auto const d = dims(mxs[0]);
        auto buffer = std::make_unique<PixType[]>(mxs.size() * d.nElems());
        for (auto i = 0u; i < mxs.size(); i++) {
            assert(mxs[i].rows() == d.rows);
            assert(mxs[i].cols() == d.cols);
            Eigen::Map<Mx>{buffer.get() + i * d.nElems(), d.rows, d.cols} = mxs[i];
        }
        return buffer;
    }

    [[nodiscard]] auto generateThresholdRange(Mx const& mx, unsigned nThresholds)
        -> std::vector<double> {
        auto const max = mx.maxCoeff();
        auto const min = mx.minCoeff();

        auto vals = std::vector<double>(nThresholds);
        std::iota(vals.begin(), vals.end(), 1.0);

        auto const delta = static_cast<double>(max - min) / nThresholds;
        std::transform(vals.begin(), vals.end(), vals.begin(), [delta, min](auto v) {
            return static_cast<double>(min) + delta * v;
        });

        return vals;
    }

    [[nodiscard]] auto predict(Mx const& mx, double threshold)
        -> std::vector<std::vector<unsigned>> {
        auto allPredictions = std::vector<std::vector<unsigned>>{};
        allPredictions.reserve(mx.cols());

        MxBool const mask = mx.array() < threshold;
        for (auto j = 0u; j < mx.cols(); j++) {
            auto predictions = std::vector<unsigned>{};
            for (auto i = 0u; i < mx.rows(); i++) {
                if (mask(i, j)) {
                    predictions.push_back(i);
                }
            }
            allPredictions.push_back(predictions);
        }
        return allPredictions;
    }

    [[nodiscard]] auto analysePredictions(std::vector<std::vector<unsigned>> const& predictions,
                                          std::vector<std::vector<unsigned>> const& truth)
        -> predictionStats {
        assert(predictions.size() == truth.size());

        auto stats = predictionStats{};
        for (auto i = 0; i < static_cast<int>(predictions.size()); i++) {
            auto tps = std::vector<unsigned>{};
            auto fps = std::vector<unsigned>{};
            auto fns = std::vector<unsigned>{};
            std::set_intersection(predictions[i].begin(),
                                  predictions[i].end(),
                                  truth[i].begin(),
                                  truth[i].end(),
                                  std::back_inserter(tps));
            std::set_difference(predictions[i].begin(),
                                predictions[i].end(),
                                truth[i].begin(),
                                truth[i].end(),
                                std::back_inserter(fps));
            std::set_difference(
                truth[i].begin(), truth[i].end(), tps.begin(), tps.end(), std::back_inserter(fns));
            stats.truePositive += tps.size();
            stats.falsePositive += fps.size();
            stats.falseNegative += fns.size();
        }
        stats.recall = static_cast<double>(stats.truePositive) /
                       static_cast<double>(stats.truePositive + stats.falseNegative);
        stats.precision = static_cast<double>(stats.truePositive) /
                          static_cast<double>(stats.truePositive + stats.falsePositive);

        return stats;
    }

    [[nodiscard]] auto prCurve(Mx const& mx,
                               std::vector<std::vector<unsigned>> const& truth,
                               unsigned nPoints) -> std::vector<predictionStats> {
        auto const thresholds = generateThresholdRange(mx, nPoints);

        auto stats = std::vector<predictionStats>{};
        stats.reserve(thresholds.size());
        std::transform(thresholds.begin(),
                       thresholds.end(),
                       std::back_inserter(stats),
                       [&mx, &truth](auto threshold) {
                           return analysePredictions(predict(mx, threshold), truth);
                       });

        return stats;
    }

    void writePrCurveToCsv(std::vector<predictionStats> const& stats,
                           std::filesystem::path const& file) {
        std::ofstream f{file};
        f << "True Positive, False Positive, False Negative, Precision, Recall\n";
        for (auto const& s : stats) {
            f << fmt::format("{}, {}, {}, {}, {}\n",
                             s.truePositive,
                             s.falsePositive,
                             s.falseNegative,
                             s.precision,
                             s.recall);
        }
    }

    [[nodiscard]] auto nordlandGroundTruth(unsigned n) -> std::vector<std::vector<unsigned>> {
        auto truth = std::vector<std::vector<unsigned>>{};
        truth.reserve(n);
        for (auto i = 0u; i < n; i++) {
            truth.push_back(std::vector<unsigned>{i});
        }
        return truth;
    }

    namespace cpu {
        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize) noexcept -> Mx {
            Mx mx = Mx(referenceMxs.size(), queryMxs.size());
            for (auto tr = 0u; tr < referenceMxs.size() / tileSize; tr++) {
                for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                    for (auto i = 0u; i < tileSize; i++) {
                        for (auto j = 0u; j < tileSize; j++) {
                            auto const r = tr * tileSize + i;
                            auto const q = tq * tileSize + j;
                            mx(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                        }
                    }
                }
            }
            auto const rrem = referenceMxs.size() % tileSize;
            auto const qrem = queryMxs.size() % tileSize;
            for (auto tr = 0u; tr < referenceMxs.size() / tileSize; tr++) {
                for (auto i = 0u; i < tileSize; i++) {
                    for (auto j = 0u; j < qrem; j++) {
                        auto const r = tr * tileSize + i;
                        auto const q = queryMxs.size() - qrem + j;
                        mx(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                    }
                }
            }
            for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                for (auto i = 0u; i < rrem; i++) {
                    for (auto j = 0u; j < tileSize; j++) {
                        auto const r = referenceMxs.size() - rrem + i;
                        auto const q = tq * tileSize + j;
                        mx(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                    }
                }
            }
            for (auto r = referenceMxs.size() - rrem; r < referenceMxs.size(); r++) {
                for (auto q = queryMxs.size() - qrem; q < queryMxs.size(); q++) {
                    mx(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                }
            }
            return mx;
        }

        [[nodiscard]] auto enhanceDiffMx(Mx const& diffMx, unsigned windowSize) noexcept -> Mx {
            Mx mx = Mx(diffMx.rows(), diffMx.cols());
            auto offset = static_cast<int>(std::floor(windowSize / 2.0));

            for (auto j = 0u; j < diffMx.cols(); ++j) {
                for (auto i = 0; i < diffMx.rows(); ++i) {
                    auto const start = static_cast<unsigned>(std::max(i - offset, 0));
                    auto const length = [i, offset, windowSize, rows = diffMx.rows()]() -> int {
                        if (i < offset) {
                            return i - offset + windowSize;
                        } else if (i - offset + windowSize > rows) {
                            return rows - i - offset + windowSize;
                        } else {
                            return windowSize;
                        }
                    }();
                    Vx const window = diffMx.block(start, j, length, 1);
                    auto const mean = window.mean();
                    auto const std = std::sqrt((window.array() - mean).pow(2).sum() / length);
                    mx(i, j) =
                        (diffMx(i, j) - mean) / std::max(std, std::numeric_limits<float>::min());
                }
            }

            return mx;
        }

        [[nodiscard]] auto sequenceSearch(Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned trajectorySteps) noexcept -> Mx {
            Mx mx = Mx::Zero(diffMx.rows(), diffMx.cols());
            auto const qi = calcTrajectoryQueryIndexOffsets(sequenceLength);
            auto const ri = calcTrajectoryReferenceIndexOffsets(qi, vMin, vMax, trajectorySteps);

            auto [qMin, qMax] = std::minmax_element(qi.begin(), qi.end());
            auto [rMin, rMax] = std::minmax_element(ri.back().begin(), ri.back().end());

            for (auto r = -*rMin; r < diffMx.rows() - *rMax; ++r) {
                for (auto q = -*qMin; q < diffMx.cols() - *qMax; ++q) {
                    auto best = std::numeric_limits<float>::max();
                    for (auto const& traj : ri) {
                        auto const score = calcTrajectoryScore(diffMx, r, q, traj, qi);
                        best = std::min(score, best);
                    }
                    mx(r, q) = best;
                }
            }
            return mx;
        }
    } // namespace cpu

    namespace opencl {
        namespace diffmxcalc {
            [[nodiscard]] auto createContext() -> clutils::Context {
                auto context = clutils::Context{};
                context.addKernels(findOrThrow(kernels.first), kernels.second);
                return context;
            }

            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             std::size_t nReference,
                                             std::size_t nQuery,
                                             std::size_t nPix) -> diffMxBuffers {
                auto r = clutils::Buffer{
                    context, nReference * nPix * sizeof(PixType), clutils::Buffer::Access::read};
                auto q = clutils::Buffer{
                    context, nQuery * nPix * sizeof(PixType), clutils::Buffer::Access::read};
                auto d = clutils::Buffer{
                    context, nReference * nQuery * sizeof(PixType), clutils::Buffer::Access::write};
                return {std::move(r), std::move(q), std::move(d)};
            }

            [[nodiscard]] constexpr auto localMemoryRequired(std::size_t nPix,
                                                             std::size_t tileSize,
                                                             std::size_t nPerThread) noexcept {
                return nPix * tileSize * tileSize * sizeof(PixType) / nPerThread;
            }

            [[nodiscard]] auto isValidParameters(std::size_t nImages,
                                                 std::size_t nPix,
                                                 std::size_t tileSize,
                                                 std::size_t nPixPerThread) noexcept -> bool {
                auto const nThreads = nPix / nPixPerThread;
                bool const fitsInLocal =
                    localMemoryRequired(nPix, tileSize, nPixPerThread) < localMemorySize;
                bool const tLessThanMax = nThreads <= 1024;
                bool const tMoreThanWarp = nThreads > 32;
                bool const tMoreThanTiles = nThreads > tileSize * tileSize;
                bool const tilesCorrectly = nImages % tileSize == 0u;
                bool const initialReduceCorrectly = nPix % nPixPerThread == 0u;
                return fitsInLocal && tLessThanMax && tMoreThanWarp && tMoreThanTiles &&
                       tilesCorrectly && initialReduceCorrectly;
            }

            void writeArgs(clutils::Context& context,
                           diffMxBuffers const& buffers,
                           std::vector<Mx> const& referenceMxs,
                           std::vector<Mx> const& queryMxs,
                           std::size_t tileSize,
                           std::size_t nPerThread,
                           std::string_view kernelName) {
                assert(!referenceMxs.empty());
                assert(!queryMxs.empty());
                assert(dims(referenceMxs[0]) == dims(queryMxs[0]));
                auto const d = dims(referenceMxs[0]);
                auto const rbuf = convertToBuffer(referenceMxs);
                auto const qbuf = convertToBuffer(queryMxs);
                buffers.reference.writeBuffer(rbuf.get());
                buffers.query.writeBuffer(qbuf.get());

                if (localMemoryRequired(d.nElems(), tileSize, nPerThread) > localMemorySize) {
                    throw std::runtime_error{fmt::format(
                        "Allocation of GPU local memory is too large with parameters number pixels "
                        "per image = {}, tile size = {}, and number of pixels per thread = {}. "
                        "Requesting memory allocation of {}, where max is {}",
                        d.nElems(),
                        tileSize,
                        nPerThread,
                        localMemoryRequired(d.nElems(), tileSize, nPerThread),
                        localMemorySize)};
                }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
                context.setKernelArg(kernelName, 0, buffers.query);
                context.setKernelArg(kernelName, 1, buffers.reference);
                context.setKernelArg(kernelName, 2, static_cast<unsigned int>(tileSize));
                context.setKernelArg(kernelName, 3, static_cast<unsigned int>(nPerThread));
                context.setKernelArg(kernelName, 4, buffers.diffMx);
                context.setKernelLocalArg(
                    kernelName, 5, d.nElems() * tileSize * tileSize * sizeof(PixType) / nPerThread);
#pragma GCC diagnostic pop
            }
        } // namespace diffmxcalc

        namespace diffmxenhance {
            [[nodiscard]] auto createContext() -> clutils::Context {
                auto context = clutils::Context{};
                context.addKernels(findOrThrow(kernels.first), kernels.second);
                return context;
            }

            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             std::size_t nReference,
                                             std::size_t nQuery) -> diffMxEnhanceBuffers {
                auto din = clutils::Buffer{
                    context, nReference * nQuery * sizeof(PixType), clutils::Buffer::Access::read};
                auto dout = clutils::Buffer{
                    context, nReference * nQuery * sizeof(PixType), clutils::Buffer::Access::write};
                return {std::move(din), std::move(dout)};
            }

            [[nodiscard]] constexpr auto localMemoryRequired(std::size_t nReference) noexcept {
                return nReference * sizeof(PixType);
            }

            [[nodiscard]] auto isValidParameters(std::size_t nReference,
                                                 unsigned nPixPerThread) noexcept -> bool {
                auto const nThreads = nReference / nPixPerThread;
                bool const fitsInLocal = localMemoryRequired(nReference) < localMemorySize;
                bool const tLessThanMax = nThreads <= 1024;
                bool const tilesCorrectly = nReference % nPixPerThread == 0u;
                return fitsInLocal && tLessThanMax && tilesCorrectly;
            }

            void writeArgs(clutils::Context& context,
                           diffMxEnhanceBuffers const& buffers,
                           Mx const& diffMx,
                           int windowSize,
                           unsigned nPixPerThread,
                           std::string_view kernelName) {
                auto in = std::make_unique<PixType[]>(diffMx.rows() * diffMx.cols());
                Eigen::Map<Mx>{in.get(), diffMx.rows(), diffMx.cols()} = diffMx;
                buffers.in.writeBuffer(in.get());

                if (localMemoryRequired(diffMx.rows()) > localMemorySize) {
                    throw std::runtime_error{fmt::format(
                        "Allocation of GPU local memory is too large with parameters number "
                        "reference images = {}.Requesting memory allocation of {}, where max is {}",
                        diffMx.rows(),
                        localMemoryRequired(diffMx.rows()),
                        localMemorySize)};
                }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
                context.setKernelArg(kernelName, 0, buffers.in);
                context.setKernelArg(kernelName, 1, static_cast<int>(windowSize));
                context.setKernelArg(kernelName, 2, static_cast<unsigned>(nPixPerThread));
                context.setKernelArg(kernelName, 3, buffers.out);
                context.setKernelLocalArg(kernelName, 4, diffMx.rows() * sizeof(PixType));
#pragma GCC diagnostic pop
            }
        } // namespace diffmxenhance

        namespace seqsearch {
            auto createContext() -> clutils::Context {
                auto context = clutils::Context{};
                context.addKernels(findOrThrow(kernels.first), kernels.second);
                return context;
            }

            [[nodiscard]] auto createBuffers(clutils::Context& context,
                                             unsigned sequenceLength,
                                             unsigned nTrajectories,
                                             std::size_t nReference,
                                             std::size_t nQuery) -> sequenceSearchBuffers {
                auto in = clutils::Buffer{
                    context, nReference * nQuery * sizeof(PixType), clutils::Buffer::Access::read};
                auto out = clutils::Buffer{
                    context, nReference * nQuery * sizeof(PixType), clutils::Buffer::Access::write};
                auto qOffsets = clutils::Buffer{
                    context, sequenceLength * sizeof(int), clutils::Buffer::Access::read};
                auto rOffsets = clutils::Buffer{context,
                                                sequenceLength * nTrajectories * sizeof(int),
                                                clutils::Buffer::Access::read};
                return {std::move(in), std::move(qOffsets), std::move(rOffsets), std::move(out)};
            }

            [[nodiscard]] auto isValidParameters(std::size_t nReference,
                                                 unsigned nPixPerThread) noexcept -> bool {
                auto const nThreads = nReference / nPixPerThread;
                bool const tLessThanMax = nThreads <= 1024;
                bool const tileCorrectly = nReference % nPixPerThread == 0u;
                return tLessThanMax && tileCorrectly;
            }

            void writeArgs(clutils::Context& context,
                           sequenceSearchBuffers const& buffers,
                           Mx const& diffMx,
                           unsigned sequenceLength,
                           float vMin,
                           float vMax,
                           unsigned nTrajectories,
                           unsigned nPixPerThread,
                           std::string_view kernelName) {
                auto const qi = calcTrajectoryQueryIndexOffsets(sequenceLength);
                auto const ri = calcTrajectoryReferenceIndexOffsets(qi, vMin, vMax, nTrajectories);

                auto in = std::make_unique<PixType[]>(diffMx.rows() * diffMx.cols());
                Eigen::Map<Mx>{in.get(), diffMx.rows(), diffMx.cols()} = diffMx;
                buffers.in.writeBuffer(in.get());

                buffers.qOffsets.writeBuffer(qi.data());
                buffers.rOffsets.writeBuffer(concatenateROffsets(ri).data());

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
                context.setKernelArg(kernelName, 0, buffers.in);
                context.setKernelArg(kernelName, 1, buffers.qOffsets);
                context.setKernelArg(kernelName, 2, buffers.rOffsets);
                context.setKernelArg(kernelName, 3, static_cast<unsigned>(qi.size()));
                context.setKernelArg(kernelName, 4, static_cast<unsigned>(ri.size()));
                context.setKernelArg(kernelName, 5, static_cast<unsigned>(nPixPerThread));
                context.setKernelArg(kernelName, 6, buffers.out);
#pragma GCC diagnostic pop
            }
        } // namespace seqsearch

        [[nodiscard]] auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize,
                                          std::size_t nPerThread,
                                          std::string_view kernelName) -> Mx {
            auto context = diffmxcalc::createContext();
            return generateDiffMx(
                context, referenceMxs, queryMxs, tileSize, nPerThread, kernelName);
        }

        [[nodiscard]] auto generateDiffMx(clutils::Context& context,
                                          std::vector<Mx> const& referenceMxs,
                                          std::vector<Mx> const& queryMxs,
                                          std::size_t tileSize,
                                          std::size_t nPerThread,
                                          std::string_view kernelName) -> Mx {
            assert(!referenceMxs.empty());
            assert(!queryMxs.empty());
            assert(dims(referenceMxs[0]) == dims(queryMxs[0]));
            auto const d = dims(referenceMxs[0]);
            auto bufs = diffmxcalc::createBuffers(
                context, referenceMxs.size(), queryMxs.size(), d.nElems());
            writeArgs(context, bufs, referenceMxs, queryMxs, tileSize, nPerThread, kernelName);

            return generateDiffMx(context,
                                  bufs.diffMx,
                                  referenceMxs.size(),
                                  queryMxs.size(),
                                  d.nElems(),
                                  tileSize,
                                  nPerThread,
                                  kernelName);
        }

        [[nodiscard]] auto generateDiffMx(clutils::Context const& context,
                                          clutils::Buffer const& outBuffer,
                                          std::size_t referenceSize,
                                          std::size_t querySize,
                                          std::size_t nPix,
                                          std::size_t tileSize,
                                          std::size_t nPerThread,
                                          std::string_view kernelName) -> Mx {
            context.runKernel(kernelName,
                              {nPix / nPerThread, referenceSize / tileSize, querySize / tileSize},
                              {nPix / nPerThread, 1, 1});

            auto buffer = std::make_unique<PixType[]>(referenceSize * querySize);
            outBuffer.readBuffer(buffer.get());

            return Mx(Eigen::Map<Mx>{buffer.get(),
                                     static_cast<Eigen::Index>(referenceSize),
                                     static_cast<Eigen::Index>(querySize)});
        }

        [[nodiscard]] auto enhanceDiffMx(Mx const& diffMx,
                                         unsigned windowSize,
                                         unsigned nPixPerThread,
                                         std::string_view kernelName) -> Mx {
            auto context = diffmxenhance::createContext();
            return enhanceDiffMx(context, diffMx, windowSize, nPixPerThread, kernelName);
        }

        [[nodiscard]] auto enhanceDiffMx(clutils::Context& context,
                                         Mx const& diffMx,
                                         unsigned windowSize,
                                         unsigned nPixPerThread,
                                         std::string_view kernelName) -> Mx {
            assert(windowSize < diffMx.rows());
            auto bufs = diffmxenhance::createBuffers(context, diffMx.rows(), diffMx.cols());
            diffmxenhance::writeArgs(context, bufs, diffMx, windowSize, nPixPerThread, kernelName);

            return enhanceDiffMx(
                context, bufs.out, diffMx.rows(), diffMx.cols(), nPixPerThread, kernelName);
        }

        [[nodiscard]] auto enhanceDiffMx(clutils::Context const& context,
                                         clutils::Buffer const& outBuffer,
                                         std::size_t nReference,
                                         std::size_t nQuery,
                                         unsigned nPixPerThread,
                                         std::string_view kernelName) -> Mx {
            context.runKernel(
                kernelName, {nReference / nPixPerThread, nQuery}, {nReference / nPixPerThread, 1});

            auto buffer = std::make_unique<PixType[]>(nReference * nQuery);
            outBuffer.readBuffer(buffer.get());

            return Mx(Eigen::Map<Mx>{buffer.get(),
                                     static_cast<Eigen::Index>(nReference),
                                     static_cast<Eigen::Index>(nQuery)});
        }

        [[nodiscard]] auto sequenceSearch(Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned nTrajectories,
                                          unsigned nPixPerThread,
                                          std::string_view kernelName) -> Mx {
            auto context = seqsearch::createContext();
            return sequenceSearch(context,
                                  diffMx,
                                  sequenceLength,
                                  vMin,
                                  vMax,
                                  nTrajectories,
                                  nPixPerThread,
                                  kernelName);
        }

        [[nodiscard]] auto sequenceSearch(clutils::Context context,
                                          Mx const& diffMx,
                                          unsigned sequenceLength,
                                          float vMin,
                                          float vMax,
                                          unsigned nTrajectories,
                                          unsigned nPixPerThread,
                                          std::string_view kernelName) -> Mx {
            auto bufs = seqsearch::createBuffers(
                context, sequenceLength, nTrajectories, diffMx.rows(), diffMx.cols());
            seqsearch::writeArgs(context,
                                 bufs,
                                 diffMx,
                                 sequenceLength,
                                 vMin,
                                 vMax,
                                 nTrajectories,
                                 nPixPerThread,
                                 kernelName);

            return sequenceSearch(
                context, bufs.out, diffMx.rows(), diffMx.cols(), nPixPerThread, kernelName);
        }

        [[nodiscard]] auto sequenceSearch(clutils::Context const& context,
                                          clutils::Buffer const& outBuffer,
                                          std::size_t nReference,
                                          std::size_t nQuery,
                                          unsigned nPixPerThread,
                                          std::string_view kernelName) -> Mx {
            context.runKernel(
                kernelName, {nReference / nPixPerThread, nQuery}, {nReference / nPixPerThread, 1});

            auto buffer = std::make_unique<PixType[]>(nReference * nQuery);
            outBuffer.readBuffer(buffer.get());

            return Mx(Eigen::Map<Mx>{buffer.get(),
                                     static_cast<Eigen::Index>(nReference),
                                     static_cast<Eigen::Index>(nQuery)});
        }
    } // namespace opencl
} // namespace seqslam
