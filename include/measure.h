#ifndef SEQSLAM_MEASURE_H
#define SEQSLAM_MEASURE_H

#include "seqslam.h"

#include <chrono>
#include <filesystem>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

namespace seqslam {
    struct seqslamParameters;
    struct result;
    struct patchWindowSize_t {};
    struct sequenceLength_t {};
    struct nTraj_t {};

    using ms = std::chrono::milliseconds;

    struct seqslamParameters {
        std::filesystem::path datasetRoot = {};
        std::filesystem::path referencePath = {};
        std::filesystem::path queryPath = {};
        unsigned imageRows = 0;
        unsigned imageCols = 0;
        unsigned imageContrastThreshold = 0;
        unsigned nQuery = 0;
        unsigned nReference = 0;
        unsigned patchWindowSize = 0;
        unsigned sequenceLength = 0;
        unsigned nTraj = 0;
        double vMin = 0.0;
        double vMax = 0.0;
    };
    struct predictionStats {
        unsigned truePositive = 0;
        unsigned falsePositive = 0;
        unsigned falseNegative = 0;
        double precision = 0.0;
        double recall = 0.0;
        double f1score = 0.0;
    };

    struct timings {
        ms diffmxcalc = {};
        ms enhancement = {};
        ms sequenceSearch = {};
        unsigned iterations = 0;
    };

    struct result {
        seqslamParameters params;
        timings times;
        std::vector<predictionStats> stats;
    };

    namespace detail {
        constexpr auto const autoSweepMinTime = ms{500};
        constexpr auto const autoSweepPrPoints = 30u;

        template <typename T>
        [[nodiscard]] auto generateRange(std::tuple<T, T, T> range) {
            auto values =
                std::vector<T>((std::get<1>(range) - std::get<0>(range)) / std::get<2>(range));
            std::iota(values.begin(), values.end(), T{0});
            std::transform(values.begin(), values.end(), values.begin(), [range](T v) {
                return v * std::get<2>(range) + std::get<0>(range);
            });
            return values;
        }

        template <typename Rep, typename Period>
        [[nodiscard]] auto generateRange(std::tuple<std::chrono::duration<Rep, Period>,
                                                    std::chrono::duration<Rep, Period>,
                                                    std::chrono::duration<Rep, Period>> range) {
            using Dur = std::chrono::duration<Rep, Period>;
            auto values =
                std::vector<Dur>((std::get<1>(range) - std::get<0>(range)) / std::get<2>(range));
            std::iota(values.begin(), values.end(), Dur{0});
            std::transform(values.begin(), values.end(), values.begin(), [range](Dur v) {
                return v * std::get<2>(range).count() + std::get<0>(range);
            });
            return values;
        }

        [[nodiscard]] auto applyRange(std::vector<seqslamParameters> const& p,
                                      std::vector<int> const& range,
                                      patchWindowSize_t) -> std::vector<seqslamParameters>;
        [[nodiscard]] auto applyRange(std::vector<seqslamParameters> const& p,
                                      std::vector<int> const& range,
                                      sequenceLength_t) -> std::vector<seqslamParameters>;
        [[nodiscard]] auto applyRange(std::vector<seqslamParameters> const& p,
                                      std::vector<int> const& range,
                                      nTraj_t) -> std::vector<seqslamParameters>;

        [[nodiscard]] auto runParameterSet(std::vector<Mx> const& referenceImages,
                                           std::vector<Mx> const& queryImages,
                                           std::vector<seqslamParameters> ps,
                                           std::vector<std::vector<unsigned>> const& groundTruth,
                                           unsigned prPoints,
                                           ms minTime) -> std::vector<result>;

        [[nodiscard]] auto findBestFromResults(std::vector<result> const& results,
                                               std::chrono::milliseconds maxTime)
            -> seqslamParameters;
    } // namespace detail

    [[nodiscard]] auto readParametersConfig() -> seqslamParameters;

    [[nodiscard]] auto readParametersConfig(std::filesystem::path const& configFile)
        -> seqslamParameters;

    [[nodiscard]] auto analysePredictions(std::vector<std::vector<unsigned>> const& predictions,
                                          std::vector<std::vector<unsigned>> const& truth)
        -> predictionStats;

    [[nodiscard]] auto prCurve(Mx const& mx,
                               std::vector<std::vector<unsigned>> const& truth,
                               unsigned nPoints) -> std::vector<predictionStats>;

    void writePrCurveToFile(seqslamParameters const& parameters,
                            std::vector<predictionStats> const& stats,
                            std::filesystem::path const& file);

    void writeResultsToFile(std::vector<result> const& results, std::filesystem::path const& file);

    void writeTimeSweepResultsToFile(std::vector<std::pair<result, ms>> const& results,
                                     std::filesystem::path const& file);

    [[nodiscard]] auto nordlandGroundTruth(unsigned n) -> std::vector<std::vector<unsigned>>;

    [[nodiscard]] auto dropFrames(std::vector<Mx> const& queryImages,
                                  std::pair<double, double> vRange,
                                  unsigned nSegments)
        -> std::pair<std::vector<Mx>, std::vector<std::vector<unsigned>>>;

    template <typename Var>
    [[nodiscard]] auto parameterSweep(std::vector<Mx> const& referenceImages,
                                      std::vector<Mx> const& queryImages,
                                      seqslamParameters const& p,
                                      std::vector<std::vector<unsigned>> const& groundTruth,
                                      unsigned prPoints,
                                      ms minTime,
                                      std::tuple<int, int, int> range,
                                      Var) {
        auto const ps = detail::applyRange(std::vector{p}, detail::generateRange(range), Var{});
        return detail::runParameterSet(
            referenceImages, queryImages, ps, groundTruth, prPoints, minTime);
    }

    template <typename Var1, typename Var2>
    [[nodiscard]] auto parameterSweep2d(std::vector<Mx> const& referenceImages,
                                        std::vector<Mx> const& queryImages,
                                        seqslamParameters const& p,
                                        std::vector<std::vector<unsigned>> const& groundTruth,
                                        unsigned prPoints,
                                        ms minTime,
                                        std::tuple<int, int, int> range1,
                                        Var1,
                                        std::tuple<int, int, int> range2,
                                        Var2) {
        auto const p1 = detail::applyRange(std::vector{p}, detail::generateRange(range1), Var1{});
        auto const p2 = detail::applyRange(p1, detail::generateRange(range2), Var2{});
        return detail::runParameterSet(
            referenceImages, queryImages, p2, groundTruth, prPoints, minTime);
    }

    template <typename Var>
    auto findOptimalParameter(std::vector<Mx> const& referenceImages,
                              std::vector<Mx> const& queryImages,
                              seqslamParameters const& p,
                              std::vector<std::vector<unsigned>> const& groundTruth,
                              ms maxTime,
                              std::tuple<int, int, int> range,
                              Var,
                              ms sweepMinTime = detail::autoSweepMinTime,
                              unsigned sweepPrPoints = detail::autoSweepPrPoints) {
        auto const results = parameterSweep(referenceImages,
                                            queryImages,
                                            p,
                                            groundTruth,
                                            sweepMinTime,
                                            sweepPrPoints,
                                            range,
                                            Var{});
        return findBestFromResults(results, maxTime);
    }

    template <typename Var>
    auto autoSweepTime(std::vector<Mx> const& referenceImages,
                       std::vector<Mx> const& queryImages,
                       seqslamParameters const& p,
                       std::vector<std::vector<unsigned>> const& groundTruth,
                       std::tuple<int, int, int> varRange,
                       Var,
                       std::tuple<ms, ms, ms> timeRange,
                       unsigned resultsPrPoints,
                       ms resultsMinTime,
                       unsigned sweepPrPoints = detail::autoSweepPrPoints,
                       ms sweepMinTime = detail::autoSweepMinTime) {
        auto const tRange = detail::generateRange(timeRange);
        auto const results = parameterSweep(referenceImages,
                                            queryImages,
                                            p,
                                            groundTruth,
                                            sweepPrPoints,
                                            sweepMinTime,
                                            varRange,
                                            Var{});
        auto timeSweepResults = std::vector<std::pair<result, ms>>{};
        for (auto t : tRange) {
            auto const optimalParams = detail::findBestFromResults(results, t);
            auto const optimalResults = detail::runParameterSet(referenceImages,
                                                                queryImages,
                                                                {optimalParams},
                                                                groundTruth,
                                                                resultsPrPoints,
                                                                resultsMinTime);
            assert(optimalResults.size() == 1u);
            timeSweepResults.push_back({optimalResults[0], t});
        }
        return timeSweepResults;
    }
} // namespace seqslam

#endif
