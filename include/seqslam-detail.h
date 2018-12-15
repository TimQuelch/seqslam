#ifndef SEQSLAM_SEQSLAM_DETAIL_H
#define SEQSLAM_SEQSLAM_DETAIL_H

#include "seqslam.h"

#include <vector>

namespace seqslam::detail {
    [[nodiscard]] auto calcTrajectoryQueryIndexOffsets(unsigned seqLength) noexcept
        -> std::vector<int>;
    [[nodiscard]] auto calcTrajectoryReferenceIndexOffsets(std::vector<int> const& qi,
                                                           float vMin,
                                                           float vMax,
                                                           unsigned nSteps) noexcept
        -> std::vector<std::vector<int>>;
    [[nodiscard]] auto calcTrajectoryScore(Mx const& diffMx,
                                           unsigned r,
                                           unsigned q,
                                           std::vector<int> const& qTrajectory,
                                           std::vector<int> const& rTrajectory) noexcept -> float;
} // namespace seqslam::detail

#endif
