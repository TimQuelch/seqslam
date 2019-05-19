#ifndef SEQSLAM_MEASURE_H
#define SEQSLAM_MEASURE_H

#include "seqslam.h"
#include "utils.h"

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

    namespace detail {
        [[nodiscard]] auto generateRange(std::pair<int, int> range) -> std::vector<int>;
        [[nodiscard]] auto applyRange(seqslamParameters const& p,
                                      std::vector<int> const& range,
                                      patchWindowSize_t) -> std::vector<seqslamParameters>;
        [[nodiscard]] auto applyRange(seqslamParameters const& p,
                                      std::vector<int> const& range,
                                      sequenceLength_t) -> std::vector<seqslamParameters>;
        [[nodiscard]] auto applyRange(seqslamParameters const& p,
                                      std::vector<int> const& range,
                                      nTraj_t) -> std::vector<seqslamParameters>;
        [[nodiscard]] auto runParameterSet(std::vector<Mx> const& referenceImages,
                                           std::vector<Mx> const& queryImages,
                                           std::vector<seqslamParameters> ps,
                                           std::vector<std::vector<unsigned>> const& groundTruth,
                                           unsigned prPoints,
                                           std::chrono::milliseconds minTime)
            -> std::vector<result>;
    } // namespace detail

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
        std::chrono::nanoseconds diffmxcalc;
        std::chrono::nanoseconds enhancement;
        std::chrono::nanoseconds sequenceSearch;
    };

    struct result {
        seqslamParameters params;
        timings times;
        std::vector<predictionStats> stats;
    };

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
                                      std::chrono::milliseconds minTime,
                                      std::pair<int, int> range,
                                      Var) {
        auto const ps = detail::applyRange(p, detail::generateRange(range), Var{});
        return detail::runParameterSet(
            referenceImages, queryImages, ps, groundTruth, prPoints, minTime);
    }
} // namespace seqslam

#endif
