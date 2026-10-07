// SPDX-License-Identifier: Apache-2.0
// Redirect only this process's SDK compiler metadata; replay is unaffected.
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace {
std::string destination(const char* name, bool& matched) {
    const char* selected = std::getenv("DSV41_SDK_METADATA_DIR");
    const char* home = std::getenv("HOME");
    if (!selected || !*selected || !home || !name) return {};
    const std::string prefix = std::string(home) + "/.habana/post_graph/" + std::to_string(getpid()) + "/";
    if (std::strncmp(name, prefix.c_str(), prefix.size())) return {};
    matched = true;
    const auto relative = std::filesystem::path(name + prefix.size());
    for (const auto& element : relative) if (element == "..") return {};
    const auto result = std::filesystem::path(selected) / std::to_string(getpid()) / relative;
    std::error_code error;
    std::filesystem::create_directories(result.parent_path(), error);
    if (error) return {};
    return result.string();
}
}

extern "C" FILE* fopen(const char* name, const char* mode) {
    static auto original = reinterpret_cast<FILE* (*)(const char*, const char*)>(dlsym(RTLD_NEXT, "fopen"));
    bool matched = false;
    const auto mapped = destination(name, matched);
    if (matched && mapped.empty()) { errno = EIO; return nullptr; }
    return original(mapped.empty() ? name : mapped.c_str(), mode);
}
extern "C" FILE* fopen64(const char* name, const char* mode) {
    static auto original = reinterpret_cast<FILE* (*)(const char*, const char*)>(dlsym(RTLD_NEXT, "fopen64"));
    bool matched = false;
    const auto mapped = destination(name, matched);
    if (matched && mapped.empty()) { errno = EIO; return nullptr; }
    return original(mapped.empty() ? name : mapped.c_str(), mode);
}
