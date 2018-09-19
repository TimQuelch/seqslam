#include "seqslam.h"

#include <filesystem>

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
} // namespace seqslam