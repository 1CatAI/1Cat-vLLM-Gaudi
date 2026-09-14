// SPDX-License-Identifier: Apache-2.0
// Shared file-backed Engram tables; only selected raw FP8 rows enter staging.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace py = pybind11;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Mapping {
    void* base_ = MAP_FAILED;
    size_t mapped_ = 0, delta_ = 0;
 public:
    Mapping(const std::string& path, uint64_t offset, size_t bytes, bool populate, bool lock) {
        require(bytes > 0, "Empty Engram mmap range");
        const auto page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        require(fd >= 0, "Cannot open the frozen Engram source file");
        struct stat info{};
        const bool valid = fstat(fd, &info) == 0 && offset <= static_cast<uint64_t>(info.st_size) &&
                           bytes <= static_cast<uint64_t>(info.st_size) - offset;
        if (!valid) { close(fd); throw std::runtime_error("Engram offsets exceed source file size"); }
        delta_ = offset % page;
        mapped_ = bytes + delta_;
        base_ = mmap(nullptr, mapped_, PROT_READ, MAP_SHARED | (populate ? MAP_POPULATE : 0), fd, offset - delta_);
        close(fd);
        require(base_ != MAP_FAILED, "Cannot mmap the Engram host shard");
        if (madvise(base_, mapped_, MADV_WILLNEED) != 0) {
            munmap(base_, mapped_); base_ = MAP_FAILED;
            throw std::runtime_error("Engram MADV_WILLNEED failed");
        }
        if (lock && mlock(base_, mapped_) != 0) {
            munmap(base_, mapped_); base_ = MAP_FAILED;
            throw std::runtime_error("Engram forced mlock failed; check RLIMIT_MEMLOCK in bytes");
        }
    }
    Mapping(const Mapping&) = delete;
    ~Mapping() { if (base_ != MAP_FAILED) munmap(base_, mapped_); }
    const uint8_t* data() const { return static_cast<const uint8_t*>(base_) + delta_; }
    size_t resident_pages() const {
        const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        constexpr size_t chunk = 65536;
        std::vector<unsigned char> status(chunk);
        size_t resident = 0;
        for (size_t begin = 0; begin < mapped_; begin += chunk * page) {
            size_t count = std::min(mapped_ - begin, chunk * page);
            require(mincore(static_cast<char*>(base_) + begin, count, status.data()) == 0, "Engram mincore failed");
            for (size_t i = 0; i < (count + page - 1) / page; ++i) resident += status[i] & 1;
        }
        return resident;
    }
};

class HostRows {
    Mapping weights_, scales_;
 public:
    const int64_t start, stop;
    const size_t width, groups;
    HostRows(const std::string& weightFile, uint64_t weightOffset, const std::string& scaleFile,
             uint64_t scaleOffset, int64_t rowStart, int64_t rowStop, size_t dimension,
             bool populate, bool forceLock)
        : weights_(weightFile, weightOffset, checked_bytes(rowStart, rowStop, dimension), populate, forceLock),
          scales_(scaleFile, scaleOffset, checked_bytes(rowStart, rowStop, dimension / 32), populate, forceLock),
          start(rowStart), stop(rowStop), width(dimension), groups(dimension / 32) {
        require(width > 0 && width % 32 == 0, "Engram FP8 scales require dimension divisible by 32");
    }
    static size_t checked_bytes(int64_t start, int64_t stop, size_t width) {
        require(start >= 0 && stop > start && width > 0 && width <= 65536, "Invalid Engram host shape");
        require(static_cast<uint64_t>(stop - start) <= SIZE_MAX / width, "Engram mmap byte count overflow");
        return static_cast<size_t>(stop - start) * width;
    }
    void gather(const int32_t* ids, size_t rows, uint8_t* weights, uint8_t* scales,
                size_t weightStride = 0, size_t scaleStride = 0) const {
        if (!weightStride) weightStride = width;
        if (!scaleStride) scaleStride = groups;
        for (size_t i = 0; i < rows; ++i) {
            int64_t row = ids[i];
            require(row >= start && row < stop, "Engram hash row does not belong to this TP head shard");
            std::memcpy(weights + i * weightStride, weights_.data() + (row - start) * width, width);
            std::memcpy(scales + i * scaleStride, scales_.data() + (row - start) * groups, groups);
        }
    }
    size_t resident_pages() const { return weights_.resident_pages() + scales_.resident_pages(); }
};

class GatherSlot : public std::enable_shared_from_this<GatherSlot> {
    enum class State { Idle, Running, Ready, Leased, Failed };
    std::mutex mutex_;
    std::condition_variable condition_;
    State state_ = State::Idle;
    bool stop_ = false;
    uint64_t generation_ = 0;
    size_t count_ = 0;
    const size_t capacity_, width_, groups_;
    std::vector<int32_t> ids_;
    std::vector<uint8_t> weights_, scales_;
    // Hold the Python array owner until the worker is joined at destruction.
    py::array packed_owner_;
    uint8_t* packed_ = nullptr;
    std::shared_ptr<HostRows> table_;
    std::string error_;
    std::thread worker_;
    long major_faults_ = 0;
    bool profiling_ = false;
    uint64_t started_ns_ = 0, finished_ns_ = 0;
    long worker_tid_ = 0;

    static uint64_t timestamp_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void work() {
        const auto worker_tid = syscall(SYS_gettid);
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return stop_ || state_ == State::Running; });
            if (stop_) return;
            auto table = table_;
            const auto count = count_;
            const bool profiling = profiling_;
            lock.unlock();
            std::string error;
            struct rusage before{}, after{};
            getrusage(RUSAGE_THREAD, &before);
            const auto started = profiling ? timestamp_ns() : 0;
            try {
                if (packed_) {
                    table->gather(ids_.data(), count, packed_, packed_ + width_, width_ + groups_, width_ + groups_);
                } else {
                    table->gather(ids_.data(), count, weights_.data(), scales_.data());
                }
            } catch (const std::exception& exc) { error = exc.what(); }
            const auto finished = profiling ? timestamp_ns() : 0;
            getrusage(RUSAGE_THREAD, &after);
            lock.lock();
            major_faults_ = after.ru_majflt - before.ru_majflt;
            started_ns_ = started;
            finished_ns_ = finished;
            worker_tid_ = worker_tid;
            error_ = std::move(error);
            state_ = error_.empty() ? State::Ready : State::Failed;
            condition_.notify_all();
        }
    }
 public:
    static size_t checked_capacity(size_t capacity, size_t width) {
        require(capacity > 0 && capacity <= 512 * 24 && width > 0 && width <= 512 && width % 32 == 0,
                "Invalid fixed Engram staging capacity");
        return capacity;
    }
    GatherSlot(size_t capacity, size_t width) : capacity_(checked_capacity(capacity, width)), width_(width), groups_(width / 32),
        ids_(capacity_), weights_(capacity_ * width_), scales_(capacity_ * groups_) {
        worker_ = std::thread([this] { work(); });
    }
    ~GatherSlot() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; condition_.notify_all(); }
        if (worker_.joinable()) worker_.join();
    }
    void bind_packed_output(py::array_t<uint8_t, py::array::c_style> output) {
        const auto info = output.request();
        std::lock_guard<std::mutex> lock(mutex_);
        require(state_ == State::Idle, "Cannot rebind an owned Engram staging buffer");
        require(output.writeable() && info.ndim == 2 && info.shape[0] == static_cast<ssize_t>(capacity_)
                && info.shape[1] == static_cast<ssize_t>(width_ + groups_),
                "Engram packed output must match its fixed row capacity and weight/scale layout");
        packed_owner_ = std::move(output);
        packed_ = static_cast<uint8_t*>(info.ptr);
    }
    uint64_t submit(std::shared_ptr<HostRows> table, py::array_t<int32_t, py::array::c_style> ids) {
        auto input = ids.request();
        std::lock_guard<std::mutex> lock(mutex_);
        require(state_ == State::Idle, "Engram staging is still owned by an earlier consumer");
        require(table && table->width == width_ && input.size > 0 && static_cast<size_t>(input.size) <= capacity_,
                "Engram row count or width exceeds the fixed staging buffer");
        count_ = input.size;
        std::memcpy(ids_.data(), input.ptr, count_ * sizeof(int32_t));
        table_ = std::move(table);
        ++generation_;
        state_ = State::Running;
        condition_.notify_all();
        return generation_;
    }
    void wait(uint64_t generation) {
        std::unique_lock<std::mutex> lock(mutex_);
        require(generation == generation_, "Stale Engram completion generation");
        condition_.wait(lock, [&] { return state_ != State::Running; });
        require(state_ == State::Ready, state_ == State::Failed ? error_.c_str() : "Engram slot has no pending result");
        state_ = State::Leased;
    }
    void release(uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(generation == generation_ && state_ == State::Leased,
                "Cannot release a stale or incomplete Engram staging generation");
        table_.reset();
        state_ = State::Idle;
    }
    py::array weights() {
        return py::array(py::dtype::of<uint8_t>(), {capacity_, width_}, {width_, size_t(1)},
                         weights_.data(), py::cast(shared_from_this()));
    }
    py::array scales() {
        return py::array(py::dtype::of<uint8_t>(), {capacity_, groups_}, {groups_, size_t(1)},
                         scales_.data(), py::cast(shared_from_this()));
    }
    long major_faults() {
        std::lock_guard<std::mutex> lock(mutex_);
        require(state_ == State::Leased, "Wait for Engram gather before reading its fault count");
        return major_faults_;
    }
    void set_profiling(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(state_ == State::Idle, "Cannot change profiling during a pending Engram gather");
        profiling_ = enabled;
    }
    std::vector<uint64_t> timing() {
        std::lock_guard<std::mutex> lock(mutex_);
        require(state_ == State::Leased && profiling_, "Engram timing requires a completed profiled gather");
        return {generation_, started_ns_, finished_ns_, count_, static_cast<uint64_t>(worker_tid_)};
    }
};
}

PYBIND11_MODULE(dsv41_host_gather, m) {
    m.attr("abi_version") = 1;
    m.attr("packed_output_version") = 1;
    py::class_<HostRows, std::shared_ptr<HostRows>>(m, "HostRows")
        .def(py::init<const std::string&, uint64_t, const std::string&, uint64_t, int64_t, int64_t, size_t, bool, bool>(),
             py::arg("weight_file"), py::arg("weight_offset"), py::arg("scale_file"), py::arg("scale_offset"),
             py::arg("row_start"), py::arg("row_stop"), py::arg("width"), py::arg("populate") = true,
             py::arg("force_lock") = false, py::call_guard<py::gil_scoped_release>())
        .def("resident_pages", &HostRows::resident_pages, py::call_guard<py::gil_scoped_release>());
    py::class_<GatherSlot, std::shared_ptr<GatherSlot>>(m, "GatherSlot")
        .def(py::init<size_t, size_t>(), py::arg("row_capacity"), py::arg("width"))
        .def("submit", &GatherSlot::submit)
        .def("bind_packed_output", &GatherSlot::bind_packed_output, py::arg("output").noconvert())
        .def("wait", &GatherSlot::wait, py::call_guard<py::gil_scoped_release>())
        .def("release", &GatherSlot::release)
        .def("set_profiling", &GatherSlot::set_profiling)
        .def_property_readonly("timing", &GatherSlot::timing)
        .def_property_readonly("weights", &GatherSlot::weights)
        .def_property_readonly("scales", &GatherSlot::scales)
        .def_property_readonly("major_faults", &GatherSlot::major_faults);
}
