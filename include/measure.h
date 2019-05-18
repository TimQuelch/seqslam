#ifndef SEQSLAM_MEASURE_H
#define SEQSLAM_MEASURE_H

#include "seqslam.h"
#include "utils.h"

#include <filesystem>
#include <vector>

#include <Eigen/Dense>

namespace seqslam {
    struct predictionStats {
        unsigned truePositive = 0;
        unsigned falsePositive = 0;
        unsigned falseNegative = 0;
        double precision = 0.0;
        double recall = 0.0;
        double f1score = 0.0;
    };

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

    [[nodiscard]] auto readParametersConfig(std::filesystem::path const& configFile)
        -> seqslamParameters;

    [[nodiscard]] auto analysePredictions(std::vector<std::vector<unsigned>> const& predictions,
                                          std::vector<std::vector<unsigned>> const& truth)
        -> predictionStats;

    [[nodiscard]] auto prCurve(Mx const& mx,
                               std::vector<std::vector<unsigned>> const& truth,
                               unsigned nPoints) -> std::vector<predictionStats>;

    void writePrCurveToJson(seqslamParameters const& parameters,
                            std::vector<predictionStats> const& stats,
                            std::filesystem::path const& file);

    [[nodiscard]] auto nordlandGroundTruth(unsigned n) -> std::vector<std::vector<unsigned>>;

    [[nodiscard]] auto dropFrames(std::vector<Mx> const& queryImages,
                                  std::pair<double, double> vRange,
                                  unsigned nSegments)
        -> std::pair<std::vector<Mx>, std::vector<std::vector<unsigned>>>;
} // namespace seqslam

#endif
