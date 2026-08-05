#pragma once

#include <spdlog/spdlog.h>

#include <string>

namespace mc_tracking::utils {

void init_logger(const std::string& level, bool json);

}  // namespace mc_tracking::utils

// These are compile-time gated by SPDLOG_ACTIVE_LEVEL, which CMakeLists.txt
// sets to TRACE on mc_tracking_core (PUBLIC). Below that level the call site
// is removed entirely, so init_logger's runtime level cannot bring it back:
// if you ever build these files outside that target, carry the definition.
#define MCT_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define MCT_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define MCT_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define MCT_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define MCT_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define MCT_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
