#include "seqslam.h"

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace seqslam {
    auto readImages(std::filesystem::path dir) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> images;
        for (const auto& imagePath : std::filesystem::directory_iterator(dir)) {
            images.push_back(cv::imread(imagePath.path().string(), CV_LOAD_IMAGE_GRAYSCALE));
        }
        return images;
    }

    auto contrastEnhancement(const std::vector<cv::Mat>& images, double threshold)
        -> std::vector<cv::Mat> {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(threshold);

        std::vector<cv::Mat> contrastEnhanced;
        std::transform(images.begin(),
                       images.end(),
                       std::back_inserter(contrastEnhanced),
                       [&clahe](const auto& im) {
                           cv::Mat res;
                           clahe->apply(im, res);
                           return res;
                       });
        return contrastEnhanced;
    }

    auto convertToEigen(const std::vector<cv::Mat>& images) -> ImgMxVector {
        ImgMxVector res;
        res.reserve(images.size());
        std::transform(images.begin(), images.end(), std::back_inserter(res), [](const auto& im) {
            ImgMx mat;
            cv::cv2eigen(im, mat);
            return mat;
        });
        return res;
    }

    auto convertToCv(const ImgMxVector& mxs) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> res;
        res.reserve(mxs.size());
        std::transform(mxs.begin(), mxs.end(), std::back_inserter(res), [](const auto& mx) {
            cv::Mat im;
            cv::eigen2cv(mx, im);
            return im;
        });
        return res;
    }

    namespace cpu {
        auto generateDiffMx(const ImgMxVector& referenceMxs,
                            const ImgMxVector& queryMxs,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            auto mx = std::make_unique<DiffMx>(referenceMxs.size(), queryMxs.size());
            for (auto tr = 0u; tr < referenceMxs.size() / tileSize; tr++) {
                for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                    for (auto i = 0u; i < tileSize; i++) {
                        for (auto j = 0u; j < tileSize; j++) {
                            const auto r = tr * tileSize + i;
                            const auto q = tq * tileSize + j;
                            (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                        }
                    }
                }
            }
            const auto rrem = referenceMxs.size() % tileSize;
            const auto qrem = queryMxs.size() % tileSize;
            for (auto tr = 0u; tr < referenceMxs.size() / tileSize; tr++) {
                for (auto i = 0u; i < tileSize; i++) {
                    for (auto j = 0u; j < qrem; j++) {
                        const auto r = tr * tileSize + i;
                        const auto q = queryMxs.size() - qrem + j;
                        (*mx)(r, q) = (referenceMxs[r] - queryMxs[q]).cwiseAbs().sum();
                    }
                }
            }
            for (auto tq = 0u; tq < queryMxs.size() / tileSize; tq++) {
                for (auto i = 0u; i < rrem; i++) {
                    for (auto j = 0u; j < tileSize; j++) {
                        const auto r = referenceMxs.size() - rrem + i;
                        const auto q = tq * tileSize + j;
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
        auto convertToBuffer(const ImgMxVector& images, Context& context, Buffer::Access access)
            -> Buffer {
            auto clbuffer =
                Buffer{context, images.size() * nRows * nCols * sizeof(PixType), access};
            auto buffer = std::make_unique<PixType[]>(images.size() * nRows * nCols);
            for (auto i = 0u; i < images.size(); i++) {
                Eigen::Map<ImgMx>{buffer.get() + i* nRows* nCols} = images[i];
            }
            clbuffer.writeBuffer(buffer.get());
            return clbuffer;
        }

        auto generateDiffMx(ImgMxVector const& reference,
                            ImgMxVector const& query,
                            std::size_t tileSize) -> std::unique_ptr<DiffMx> {
            auto context = opencl::Context{};

            auto referenceBuffer =
                convertToBuffer(reference, context, opencl::Buffer::Access::read);
            auto queryBuffer = convertToBuffer(query, context, opencl::Buffer::Access::read);
            auto resultBuffer = opencl::Buffer{context,
                                               reference.size() * query.size() * sizeof(PixType),
                                               opencl::Buffer::Access::write};

            context.addKernels("kernels/diff-mx.cl", {"diffMx"});
            context.setKernelArg("diffMx", 0, queryBuffer);
            context.setKernelArg("diffMx", 1, referenceBuffer);
            context.setKernelArg("diffMx", 2, static_cast<std::size_t>(nRows * nCols));
            context.setKernelArg("diffMx", 3, static_cast<std::size_t>(tileSize));
            context.setKernelArg("diffMx", 4, resultBuffer);
            context.setKernelLocalArg(
                "diffMx", 5, nRows * nCols * tileSize * tileSize * sizeof(PixType));

            context.runKernel("diffMx",
                              {query.size() / tileSize, reference.size() / tileSize, nRows * nCols},
                              {1, 1, nRows * nCols});

            auto buffer = std::make_unique<PixType[]>(reference.size() * query.size());
            resultBuffer.readBuffer(buffer.get());

            DiffMx mx = Eigen::Map<DiffMx>{buffer.get(),
                                           static_cast<Eigen::Index>(reference.size()),
                                           static_cast<Eigen::Index>(query.size())};

            return std::make_unique<DiffMx>(mx);
        }
    } // namespace opencl
} // namespace seqslam
