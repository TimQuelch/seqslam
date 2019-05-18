#include "measure.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>

#include <nlohmann/json.hpp>

namespace seqslam {
    namespace detail {
        auto rng = std::mt19937{42};

        constexpr auto const tr = 1u;
        constexpr auto const tq = 24u;
        constexpr auto const nPixPerThread = 8u;

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

        [[nodiscard]] auto generateRange(std::pair<int, int> range) {
            auto values = std::vector<int>(range.second - range.first + 1);
            std::iota(values.begin(), values.end(), range.first);
            return values;
        }

        [[nodiscard]] auto
        applyRange(seqslamParameters const& p, std::vector<int> const& range, patchWindowSize_t) {
            auto ps = std::vector<seqslamParameters>{};
            std::transform(
                range.begin(), range.end(), std::back_inserter(ps), [p](auto windowSize) {
                    auto newP = p;
                    newP.patchWindowSize = windowSize;
                    return newP;
                });
            return ps;
        }

        [[nodiscard]] auto
        applyRange(seqslamParameters const& p, std::vector<int> const& range, sequenceLength_t) {
            auto ps = std::vector<seqslamParameters>{};
            std::transform(
                range.begin(), range.end(), std::back_inserter(ps), [p](auto sequenceLength) {
                    auto newP = p;
                    newP.sequenceLength = sequenceLength;
                    return newP;
                });
            return ps;
        }

        [[nodiscard]] auto
        applyRange(seqslamParameters const& p, std::vector<int> const& range, nTraj_t) {
            auto ps = std::vector<seqslamParameters>{};
            std::transform(range.begin(), range.end(), std::back_inserter(ps), [p](auto nTraj) {
                auto newP = p;
                newP.nTraj = nTraj;
                return newP;
            });
            return ps;
        }

        [[nodiscard]] auto runAndAnalyse(std::vector<Mx> const& referenceImages,
                                         std::vector<Mx> const& queryImages,
                                         seqslamParameters const& p,
                                         std::vector<std::vector<unsigned>> const& groundTruth,
                                         unsigned nPoints) {
            auto const diffMatrix =
                opencl::generateDiffMx(referenceImages, queryImages, tr, tq, nPixPerThread);
            auto const enhanced = opencl::enhanceDiffMx(diffMatrix, p.patchWindowSize);
            auto const sequences =
                opencl::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
            return prCurve(sequences, groundTruth, nPoints);
        }

        [[nodiscard]] auto runAndTime(std::vector<Mx> const& referenceImages,
                                      std::vector<Mx> const& queryImages,
                                      seqslamParameters const& p,
                                      std::chrono::milliseconds minTime) {
            using hrclock = std::chrono::high_resolution_clock;
            auto const start = hrclock::now();

            auto diffmxTimes = std::vector<hrclock::duration>{};
            auto enhanceTimes = std::vector<hrclock::duration>{};
            auto searchTimes = std::vector<hrclock::duration>{};

            while (hrclock::now() - start < minTime) {
                auto const begin = hrclock::now();
                auto const diffMatrix =
                    opencl::generateDiffMx(referenceImages, queryImages, tr, tq, nPixPerThread);
                auto const postdiffmx = hrclock::now();
                auto const enhanced = opencl::enhanceDiffMx(diffMatrix, p.patchWindowSize);
                auto const postenhanced = hrclock::now();
                auto const sequences =
                    opencl::sequenceSearch(enhanced, p.sequenceLength, p.vMin, p.vMax, p.nTraj);
                auto const end = hrclock::now();

                diffmxTimes.push_back(postdiffmx - begin);
                enhanceTimes.push_back(postenhanced - postdiffmx);
                searchTimes.push_back(end - postenhanced);
            }

            return timings{
                std::chrono::duration_cast<std::chrono::nanoseconds>(hrclock::duration{
                    std::accumulate(diffmxTimes.begin(), diffmxTimes.end(), hrclock::duration{0}) /
                    diffmxTimes.size()}),
                std::chrono::duration_cast<std::chrono::nanoseconds>(hrclock::duration{
                    std::accumulate(
                        enhanceTimes.begin(), enhanceTimes.end(), hrclock::duration{0}) /
                    enhanceTimes.size()}),
                std::chrono::duration_cast<std::chrono::nanoseconds>(hrclock::duration{
                    std::accumulate(searchTimes.begin(), searchTimes.end(), hrclock::duration{0}) /
                    searchTimes.size()})};
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

    [[nodiscard]] auto runParameterSet(std::vector<Mx> const& referenceImages,
                                       std::vector<Mx> const& queryImages,
                                       std::vector<seqslamParameters> ps,
                                       std::vector<std::vector<unsigned>> const& groundTruth,
                                       unsigned prPoints,
                                       std::chrono::milliseconds minTime) {
        auto results = std::vector<result>{};
        std::transform(
            ps.begin(),
            ps.end(),
            std::back_inserter(results),
            [&referenceImages, &queryImages, &groundTruth, prPoints, minTime](auto const& p) {
                return result{
                    p,
                    detail::runAndTime(referenceImages, queryImages, p, minTime),
                    detail::runAndAnalyse(referenceImages, queryImages, p, groundTruth, prPoints)};
            });

        return 0; // TODO: Do something with the results
    }

    template <typename Var>
    [[nodiscard]] auto parameterSweep(std::vector<Mx> const& referenceImages,
                                      std::vector<Mx> const& queryImages,
                                      seqslamParameters const& p,
                                      std::vector<std::vector<unsigned>> const& groundTruth,
                                      unsigned prPoints,
                                      std::chrono::milliseconds minTime,
                                      std::pair<int, int> range,
                                      Var) {
        auto const ps = detail::applyRange(p, detail::generateRange(range), Var{});
        runAndTime(referenceImages, queryImages, ps, groundTruth, prPoints, minTime);
    }
} // namespace seqslam
