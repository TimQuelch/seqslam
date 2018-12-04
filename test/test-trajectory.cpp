#include <catch2/catch.hpp>

#include "seqslam.h"
#include "seqslam-detail.h"

TEST_CASE("Query offsets calculated correctly", "[trajectory]") {
    auto const qi1 = seqslam::detail::calcTrajectoryQueryIndexOffsets(3u);

    REQUIRE(qi1.size() == 3);
    REQUIRE(qi1[0] == -1);
    REQUIRE(qi1[1] == 0);
    REQUIRE(qi1[2] == 1);

    auto const qi2 = seqslam::detail::calcTrajectoryQueryIndexOffsets(4u);

    REQUIRE(qi2.size() == 4);
    REQUIRE(qi2[0] == -2);
    REQUIRE(qi2[1] == -1);
    REQUIRE(qi2[2] == 0);
    REQUIRE(qi2[3] == 1);

    auto const qi3 = seqslam::detail::calcTrajectoryQueryIndexOffsets(9u);

    REQUIRE(qi3.size() == 9);
    REQUIRE(qi3.front() == -4);
    REQUIRE(qi3.back() == 4);
}

TEST_CASE("Reference offsets calculated correctly", "[trajectory]") {
    auto const vMax = 5.0;
    auto const seqLength = 5u;
    auto const nTraj = 3u;

    auto const qi = seqslam::detail::calcTrajectoryQueryIndexOffsets(seqLength);
    auto const ri = seqslam::detail::calcTrajectoryReferenceIndexOffsets(qi, 0, vMax, nTraj);

    REQUIRE(ri.size() == 3);

    for (auto const& traj : ri) {
        REQUIRE(traj.size() == qi.size());
    }

    for (auto i = 0u; i < qi.size(); i++) {
        REQUIRE(ri.front()[i] == 0);
        REQUIRE(ri.back()[i] == std::round(vMax * qi[i]));
    }
}
