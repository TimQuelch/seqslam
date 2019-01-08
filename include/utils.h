#ifndef SEQSLAM_UTILS_H
#define SEQSLAM_UTILS_H

#include <filesystem>
#include <optional>

namespace utils {
    [[nodiscard]] auto traverseUpUntilMatch(std::filesystem::path const& suffix)
        -> std::optional<std::filesystem::path>;
}

#endif
