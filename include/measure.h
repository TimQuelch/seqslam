#ifndef SEQSLAM_MEASURE_H
#define SEQSLAM_MEASURE_H

#include "seqslam.h"

#include <chrono>
#include <filesystem>
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
        constexpr auto const autoSweepMinTime = ms{100};
        constexpr auto const autoSweepPrPoints = 30u;

        [[nodiscard]] auto generateRange(std::tuple<int, int, int> range) -> std::vector<int>;
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
                                           std::chrono::milliseconds minTime)
            -> std::vector<result>;

        [[nodiscard]] auto findBestFromResults(std::vector<result> const& results,
                                               std::chrono::milliseconds maxTime)
            -> seqslamParameters;

        [[nodiscard]] constexpr auto extractFromResult(seqslamParameters const& p,
                                                       patchWindowSize_t) {
            return p.patchWindowSize;
        }

        [[nodiscard]] constexpr auto extractFromResult(seqslamParameters const& p,
                                                       sequenceLength_t) {
            return p.sequenceLength;
        }

        [[nodiscard]] constexpr auto extractFromResult(seqslamParameters const& p, nTraj_t) {
            return p.nTraj;
        }
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

    void writeResultsToFile(std::vector<result> const& stats, std::filesystem::path const& file);

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
                                        std::chrono::milliseconds minTime,
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
                              Var) {
        auto const results = parameterSweep(referenceImages,
                                            queryImages,
                                            p,
                                            groundTruth,
                                            detail::autoSweepPrPoints,
                                            detail::autoSweepMinTime,
                                            range,
                                            Var{});
        auto const best = findBestFromResults(results, maxTime);
        return extractFromParameters(best, Var{});
    }
} // namespace seqslam

#endif
