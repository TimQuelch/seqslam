#ifndef SEQSLAM_SEQSLAM_DETAIL_H
#define SEQSLAM_SEQSLAM_DETAIL_H

#include "seqslam.h"

#include <vector>

namespace seqslam::detail {
    auto calcTrajectoryQueryIndexOffsets(unsigned seqLength) -> std::vector<int>;
    auto calcTrajectoryReferenceIndexOffsets(std::vector<int> const& qi,
                                             float vMin,
                                             float vMax,
                                             unsigned nSteps) -> std::vector<std::vector<int>>;
    auto calcTrajectoryScore(Mx const& diffMx,
                             unsigned r,
                             unsigned q,
                             std::vector<int> const& qTrajectory,
                             std::vector<int> const& rTrajectory) -> float;
} // namespace seqslam::detail

#endif
