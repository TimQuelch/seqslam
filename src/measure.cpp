#include "measure.h"

#include <algorithm>
#include <fstream>
#include <random>

#include <nlohmann/json.hpp>

namespace seqslam {
    namespace detail {
        auto rng = std::mt19937{42};

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
    } // namespace detail

    [[nodiscard]] auto readParametersConfig(std::filesystem::path const& configFile)
        -> seqslamParameters {
        auto const json =
            utils::readJsonConfig(utils::traverseUpUntilMatch(configFile).value())["seqslam"];
        auto p = seqslamParameters{};
        p.datasetRoot = json["Dataset root path"].get<std::string>();
        p.queryPath = json["Query image path"].get<std::string>();
        p.referencePath = json["Reference image path"].get<std::string>();
        p.imageRows = json["Image rows"];
        p.imageCols = json["Image columns"];
        p.imageContrastThreshold = json["Image contrast enhancement threshold"];
        p.patchWindowSize = json["Patch window size"];
        p.sequenceLength = json["Sequence length"];
        p.vMin = json["Min velocity"];
        p.vMax = json["Max velocity"];
        p.nTraj = json["Number of trajectories"];

        return p;
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
        stats.f1score = 2 * stats.precision * stats.recall / (stats.precision + stats.recall);

        return stats;
    }

    [[nodiscard]] auto prCurve(Mx const& mx,
                               std::vector<std::vector<unsigned>> const& truth,
                               unsigned nPoints) -> std::vector<predictionStats> {
        auto const thresholds = detail::generateThresholdRange(mx, nPoints);

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

    void writePrCurveToJson(seqslamParameters const& parameters,
                            std::vector<predictionStats> const& stats,
                            std::filesystem::path const& file) {
        auto j = nlohmann::json{{"parameters",
                                 {{"Image rows", parameters.imageRows},
                                  {"Image columns", parameters.imageCols},
                                  {"Number of query images", parameters.nQuery},
                                  {"Nubmer of reference images", parameters.nReference},
                                  {"Patch window size", parameters.patchWindowSize},
                                  {"Sequence length", parameters.sequenceLength},
                                  {"Number of trajectories", parameters.nTraj},
                                  {"Min velocity", parameters.vMin},
                                  {"Max velocity", parameters.vMax}}}};

        std::transform(
            stats.begin(), stats.end(), std::back_inserter(j["data"]), [](auto const& s) {
                return nlohmann::json{{"True Positive", s.truePositive},
                                      {"False Positive", s.falsePositive},
                                      {"False Negative", s.falseNegative},
                                      {"Precision", s.precision},
                                      {"Recall", s.recall},
                                      {"F1 Score", s.f1score}};
            });

        std::ofstream f{file};
        f << j.dump(4);
    }

    [[nodiscard]] auto nordlandGroundTruth(unsigned n) -> std::vector<std::vector<unsigned>> {
        auto truth = std::vector<std::vector<unsigned>>{};
        truth.reserve(n);
        for (auto i = 0u; i < n; i++) {
            truth.push_back(std::vector<unsigned>{i});
        }
        return truth;
    }

    [[nodiscard]] auto dropFrames(std::vector<Mx> const& queryImages,
                                  std::pair<double, double> vRange,
                                  unsigned nSegments)
        -> std::pair<std::vector<Mx>, std::vector<std::vector<unsigned>>> {
        auto const intervals = [vRange, nSegments]() {
            auto vs = std::vector<double>{};
            std::generate_n(std::back_inserter(vs), nSegments, [vRange]() {
                return std::uniform_real_distribution{vRange.first, vRange.second}(detail::rng);
            });
            auto is = std::vector<unsigned>{};
            std::transform(vs.begin(), vs.end(), std::back_inserter(is), [](auto v) {
                return std::round(v / (v - 1));
            });
            return is;
        }();

        auto const segmentLength =
            std::ceil(static_cast<double>(queryImages.size()) / static_cast<double>(nSegments));

        auto mask = std::vector<bool>{};
        for (auto i = 0u; i < nSegments; i++) {
            for (auto j = 0u; j < segmentLength; j++) {
                mask.push_back(j % intervals[i] != 0);
            }
        }

        assert(mask.size() >= queryImages.size());

        auto ret = std::vector<Mx>{};
        auto const oldGroundTruth = nordlandGroundTruth(queryImages.size());
        auto newGroundTruth = decltype(oldGroundTruth){};
        for (auto i = 0u; i < queryImages.size(); i++) {
            if (mask[i]) {
                ret.push_back(queryImages[i]);
                newGroundTruth.push_back(oldGroundTruth[i]);
            }
        }

        return {ret, newGroundTruth};
    }
} // namespace seqslam
