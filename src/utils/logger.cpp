#include "mc_tracking/utils/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <unordered_map>

namespace mc_tracking::utils {

namespace {

spdlog::level::level_enum parse_level(const std::string& level) {
    static const std::unordered_map<std::string, spdlog::level::level_enum> table = {
        {"trace", spdlog::level::trace},     {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},       {"warn", spdlog::level::warn},
        {"warning", spdlog::level::warn},    {"error", spdlog::level::err},
        {"err", spdlog::level::err},         {"critical", spdlog::level::critical},
        {"off", spdlog::level::off},
    };
    const auto it = table.find(level);
    if (it == table.end()) throw std::invalid_argument("unknown log level: " + level);
    return it->second;
}

}  // namespace

void init_logger(const std::string& level, bool json) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("mc_tracking", sink);
    if (json) {
        logger->set_pattern(
            R"({"ts":"%Y-%m-%dT%H:%M:%S.%e%z","level":"%l","logger":"%n","msg":"%v"})");
    } else {
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    }
    logger->set_level(parse_level(level));
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);
}

}  // namespace mc_tracking::utils
