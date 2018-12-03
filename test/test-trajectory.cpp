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
