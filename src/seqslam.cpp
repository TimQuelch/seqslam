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

    auto convertToEigen(const std::vector<cv::Mat>& images) -> imgMxVector {
        imgMxVector res;
        res.reserve(images.size());
        std::transform(images.begin(), images.end(), std::back_inserter(res), [](const auto& im) {
            imgMx mat;
            cv::cv2eigen(im, mat);
            return mat;
        });
        return res;
    }

    auto convertToCv(const imgMxVector& mxs) -> std::vector<cv::Mat> {
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
        auto generateDiffMx(const imgMxVector& referenceMats, const imgMxVector& queryMats)
            -> diffMx {
            diffMx mx{referenceMats.size(), queryMats.size()};
            for (auto i = 0u; i < referenceMats.size(); i++) {
                for (auto j = 0u; j < queryMats.size(); j++) {
                    mx(i, j) = (referenceMats[i] - queryMats[j]).cwiseAbs().sum();
                }
            }
            mx.array() -= mx.minCoeff();
            mx /= mx.maxCoeff();
            return mx;
        }

        auto enhanceDiffMxContrast(const diffMx& mx) -> diffMx {
            diffMx res = diffMx{mx.rows(), mx.cols()};
            for (auto i = 0u; i < mx.cols(); i++) {
                auto mean = mx.col(i).mean();
                auto std = std::sqrt((mx.col(i).array() - mean).square().sum() / (mx.rows() - 1));
                res.col(i) = (mx.col(i).array() - mean) / std;
            }
            res.array() -= res.minCoeff();
            res /= res.maxCoeff();
            return res;
        }
    } // namespace cpu
} // namespace seqslam
