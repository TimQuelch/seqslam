#ifndef SEQSLAM_UTILS_H
#define SEQSLAM_UTILS_H

#include <filesystem>
#include <optional>

#include <nlohmann/json.hpp>

namespace utils {
    constexpr auto const defaultConfig = std::string_view{"config.json"};

    [[nodiscard]] auto traverseUpUntilMatch(std::filesystem::path const& suffix)
        -> std::optional<std::filesystem::path>;

    [[nodiscard]] auto readJsonConfig(std::filesystem::path const& configFile) -> nlohmann::json;
} // namespace utils

#endif
