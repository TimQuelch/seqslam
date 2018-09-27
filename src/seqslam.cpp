#include "seqslam.h"

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace seqslam {
    auto readImages(std::filesystem::path dir) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> images;
        for (auto const& imagePath : std::filesystem::directory_iterator(dir)) {
            images.push_back(cv::imread(imagePath.path().string(), CV_LOAD_IMAGE_GRAYSCALE));
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

    auto convertToEigen(std::vector<cv::Mat> const& images) -> ImgMxVector {
        ImgMxVector res;
        res.reserve(images.size());
        std::transform(images.begin(), images.end(), std::back_inserter(res), [](auto const& im) {
            ImgMx mat;
            cv::cv2eigen(im, mat);
            return mat;
        });
        return res;
    }

    auto convertToCv(ImgMxVector const& mxs) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> res;
        res.reserve(mxs.size());
        std::transform(mxs.begin(), mxs.end(), std::back_inserter(res), [](auto const& mx) {
            cv::Mat im;
            cv::eigen2cv(mx, im);
            return im;
        });
        return res;
    }

    namespace cpu {
        auto generateDiffMx(ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            auto mx = std::make_unique<DiffMx>(referenceMxs.size(), queryMxs.size());
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
        auto convertToBuffer(ImgMxVector const& images,
                             Context const& context,
                             Buffer::Access access) -> Buffer {
            auto clbuffer =
                Buffer{context, images.size() * nRows * nCols * sizeof(PixType), access};
            auto buffer = std::make_unique<PixType[]>(images.size() * nRows * nCols);
            for (auto i = 0u; i < images.size(); i++) {
                Eigen::Map<ImgMx>{buffer.get() + i* nRows* nCols} = images[i];
            }
            clbuffer.writeBuffer(buffer.get());
            return clbuffer;
        }

        auto generateDiffMx(ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            auto context = opencl::Context{};
            context.addKernels("kernels/diff-mx.cl", {"diffMx"});
            return generateDiffMx(context, referenceMxs, queryMxs, tileSize);
        }

        auto generateDiffMx(Context& context,
                            ImgMxVector const& referenceMxs,
                            ImgMxVector const& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            auto referenceBuffer =
                convertToBuffer(referenceMxs, context, opencl::Buffer::Access::read);
            auto queryBuffer = convertToBuffer(queryMxs, context, opencl::Buffer::Access::read);
            auto resultBuffer =
                opencl::Buffer{context,
                               referenceMxs.size() * queryMxs.size() * sizeof(PixType),
                               opencl::Buffer::Access::write};

            context.setKernelArg("diffMx", 0, queryBuffer);
            context.setKernelArg("diffMx", 1, referenceBuffer);
            context.setKernelArg("diffMx", 2, static_cast<std::size_t>(nRows * nCols));
            context.setKernelArg("diffMx", 3, static_cast<std::size_t>(tileSize));
            context.setKernelArg("diffMx", 4, resultBuffer);
            context.setKernelLocalArg(
                "diffMx", 5, nRows * nCols * tileSize * tileSize * sizeof(PixType));

            return generateDiffMx(
                context, resultBuffer, referenceMxs.size(), queryMxs.size(), tileSize);
        }

        auto generateDiffMx(Context const& context,
                            Buffer const& outBuffer,
                            std::size_t referenceSize,
                            std::size_t querySize,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            context.runKernel("diffMx",
                              {querySize / tileSize, referenceSize / tileSize, nRows * nCols},
                              {1, 1, nRows * nCols});

            auto buffer = std::make_unique<PixType[]>(referenceSize * querySize);
            outBuffer.readBuffer(buffer.get());

            DiffMx mx = Eigen::Map<DiffMx>{buffer.get(),
                                           static_cast<Eigen::Index>(referenceSize),
                                           static_cast<Eigen::Index>(querySize)};

            return std::make_unique<DiffMx>(std::move(mx));
        }
    } // namespace opencl
} // namespace seqslam
