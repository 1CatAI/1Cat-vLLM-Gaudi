// SPDX-License-Identifier: Apache-2.0
// Intel 1.16 profiler SDK registration forwarded to the current logger ABI.
#include <vector>
#include <utility>
#include <hl_logger/impl/hllog_fmt_headers.hpp>
#include <hl_logger/impl/hllog_internal_api.hpp>

namespace hl_logger::internal::v1_0 {
ResourceGuard registerLazyLogsHandler(LazyLogsHandler handler, std::string_view module);

ResourceGuard registerLazyLogsHandler(LazyLogsHandler handler) {
    return registerLazyLogsHandler(std::move(handler), "tp2_profiler_1_16");
}
}
