#include "seqslam.h"

#include <algorithm>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace seqslam {
    namespace {
        struct Dims {
            int rows;
            int cols;
            constexpr auto nElems() const { return rows * cols; }
        };

        auto dims(Mx const& mx) {
            return Dims{static_cast<int>(mx.rows()), static_cast<int>(mx.cols())};
        }

        auto dims(cv::Mat const& mx) {
            return Dims{static_cast<int>(mx.rows), static_cast<int>(mx.cols)};
        }

        [[maybe_unused]] auto operator==(Dims const& one, Dims const& two) {
            return one.rows == two.rows && one.cols == two.cols;
        }

        auto localMemoryRequired(std::size_t nPix, std::size_t tileSize, std::size_t nPerThread) {
            return nPix * tileSize * tileSize * sizeof(PixType) / nPerThread;
        }
    } // namespace

    auto readImages(std::filesystem::path const& dir) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> images;
        for (auto const& imagePath : std::filesystem::directory_iterator(dir)) {
            images.push_back(cv::imread(imagePath.path().string(), cv::IMREAD_GRAYSCALE));
        }
        return images;
    }

    auto contrastEnhancement(std::vector<cv::Mat> const& images, double threshold)
        -> std::vector<cv::Mat> {
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

    auto convertToEigen(std::vector<cv::Mat> const& images) -> std::vector<Mx> {
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

    auto convertToCv(std::vector<Mx> const& mxs) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> res;
        res.reserve(mxs.size());
        std::transform(mxs.begin(), mxs.end(), std::back_inserter(res), [](auto const& mx) {
            cv::Mat im;
            cv::eigen2cv(mx, im);
            return im;
        });
        return res;
    }

    auto convertToBuffer(std::vector<Mx> const& mxs) -> std::unique_ptr<PixType[]> {
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

    namespace cpu {
        auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<Mx> {
            auto mx = std::make_unique<Mx>(referenceMxs.size(), queryMxs.size());
            for (auto tr = 0u; tr < referenceMxs.size() / tileSize; tr++) {
                for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                    for (auto i = 0u; i < tileSize; i++) {
                        for (auto j = 0u; j < tileSize; j++) {
                            auto const r = tr * tileSize + i;
                            auto const q = tq * tileSize + j;
                            (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
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
                        (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                    }
                }
            }
            for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                for (auto i = 0u; i < rrem; i++) {
                    for (auto j = 0u; j < tileSize; j++) {
                        auto const r = referenceMxs.size() - rrem + i;
                        auto const q = tq * tileSize + j;
                        (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                    }
                }
            }
            for (auto r = referenceMxs.size() - rrem; r < referenceMxs.size(); r++) {
                for (auto q = queryMxs.size() - qrem; q < queryMxs.size(); q++) {
                    (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                }
            }
            return mx;
        }
    } // namespace cpu

    namespace opencl {
        auto createDiffMxContext() -> clutils::Context {
            auto context = clutils::Context{};
            context.addKernels(diffMatrixPath, kernelNames);
            return context;
        }

        auto createBuffers(clutils::Context& context,
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

        auto fitsInLocalMemory(std::size_t nPix, std::size_t tileSize, std::size_t nPerThread)
            -> bool {
            auto const localMemorySize = 48u * 1024u;
            return localMemoryRequired(nPix, tileSize, nPerThread) < localMemorySize;
        }

        auto isValidParameters(std::size_t nImages,
                               std::size_t nPix,
                               std::size_t tileSize,
                               std::size_t nPixPerThread) -> bool {
            auto const nThreads = nPix / nPixPerThread;
            bool const fitsInLocal = opencl::fitsInLocalMemory(nPix, tileSize, nPixPerThread);
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
                       std::string const& kernelName) {
            assert(!referenceMxs.empty());
            assert(!queryMxs.empty());
            assert(dims(referenceMxs[0]) == dims(queryMxs[0]));
            auto const d = dims(referenceMxs[0]);
            auto const rbuf = convertToBuffer(referenceMxs);
            auto const qbuf = convertToBuffer(queryMxs);
            buffers.reference.writeBuffer(rbuf.get());
            buffers.query.writeBuffer(qbuf.get());

            if (!fitsInLocalMemory(d.nElems(), tileSize, nPerThread)) {
                throw std::runtime_error{fmt::format(
                    "Allocation of GPU local memory is too large with parameters number pixels per "
                    "image = {}, tile size = {}, and number of pixels per thread = {}. Requesting "
                    "memory allocation of {}, where max is {}",
                    d.nElems(),
                    tileSize,
                    nPerThread,
                    localMemoryRequired(d.nElems(), tileSize, nPerThread),
                    48 * 1024)};
            }

            context.setKernelArg(kernelName, 0, buffers.query);
            context.setKernelArg(kernelName, 1, buffers.reference);
            context.setKernelArg(kernelName, 2, static_cast<unsigned int>(tileSize));
            context.setKernelArg(kernelName, 3, static_cast<unsigned int>(nPerThread));
            context.setKernelArg(kernelName, 4, buffers.diffMx);
            context.setKernelLocalArg(
                kernelName, 5, d.nElems() * tileSize * tileSize * sizeof(PixType) / nPerThread);
        }

        auto generateDiffMx(std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize,
                            std::size_t nPerThread,
                            std::string const& kernelName) -> std::unique_ptr<Mx> {
            auto context = createDiffMxContext();
            return generateDiffMx(
                context, referenceMxs, queryMxs, tileSize, nPerThread, kernelName);
        }

        auto generateDiffMx(clutils::Context& context,
                            std::vector<Mx> const& referenceMxs,
                            std::vector<Mx> const& queryMxs,
                            std::size_t tileSize,
                            std::size_t nPerThread,
                            std::string const& kernelName) -> std::unique_ptr<Mx> {
            assert(!referenceMxs.empty());
            assert(!queryMxs.empty());
            assert(dims(referenceMxs[0]) == dims(queryMxs[0]));
            auto const d = dims(referenceMxs[0]);
            auto bufs = createBuffers(context, referenceMxs.size(), queryMxs.size(), d.nElems());
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

        auto generateDiffMx(clutils::Context const& context,
                            clutils::Buffer const& outBuffer,
                            std::size_t referenceSize,
                            std::size_t querySize,
                            std::size_t nPix,
                            std::size_t tileSize,
                            std::size_t nPerThread,
                            std::string const& kernelName) -> std::unique_ptr<Mx> {
            context.runKernel(kernelName,
                              {nPix / nPerThread, querySize / tileSize, referenceSize / tileSize},
                              {nPix / nPerThread, 1, 1});

            auto buffer = std::make_unique<PixType[]>(referenceSize * querySize);
            outBuffer.readBuffer(buffer.get());

            return std::make_unique<Mx>(Eigen::Map<Mx>{buffer.get(),
                                                       static_cast<Eigen::Index>(referenceSize),
                                                       static_cast<Eigen::Index>(querySize)});
        }
    } // namespace opencl
} // namespace seqslam
