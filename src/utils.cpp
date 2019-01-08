#include "utils.h"

#include <stdexcept>

namespace utils {
    [[nodiscard]] auto traverseUpUntilMatch(std::filesystem::path const& suffix)
        -> std::optional<std::filesystem::path> {
        auto current = std::filesystem::current_path();
        while (!std::filesystem::exists(current / suffix)) {
            if (current == current.parent_path()) { // Reached root
                return std::nullopt;
            }
            current = current.parent_path();
        }
        return current / suffix;
    }
} // namespace utils
