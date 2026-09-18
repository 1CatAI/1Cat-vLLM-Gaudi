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
    void gather_packed(const int32_t* ids, size_t rows, uint8_t* output) const {
        for (size_t i = 0; i < rows; ++i)
            require(ids[i] >= start && ids[i] < stop, "Engram hash row is outside the bound TP table");
        for (size_t i = 0; i < rows; ++i) {
            const size_t row = ids[i] - start;
            std::memcpy(output + i * (width + groups), weights_.data() + row * width, width);
            std::memcpy(output + i * (width + groups) + width, scales_.data() + row * groups, groups);
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
    // Hold the Python owner until the worker has stopped using the buffer.
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
                    table->gather(ids_.data(), count, packed_, packed_ + width_,
                                  width_ + groups_, width_ + groups_);
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
        // A normal vLLM prefill transaction contains as many as 8192 tokens.
        // Capacity counts token/head rows, so a TP2 shard needs C8192 x 12.
        // Keep a hard upper bound and validate all following products rather
        // than retaining the historical context-512 ceiling.
        require(capacity > 0 && capacity <= 8192 * 24 && width > 0 && width <= 512 && width % 32 == 0
                    && capacity <= SIZE_MAX / width && capacity <= SIZE_MAX / (width / 32),
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
        require(output.writeable() && info.ndim == 2
                && info.shape[0] == static_cast<ssize_t>(capacity_)
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
        require(state_ == State::Leased && profiling_,
                "Engram timing requires a completed profiled gather");
        return {generation_, started_ns_, finished_ns_, count_,
                static_cast<uint64_t>(worker_tid_)};
    }
};

// C1 owns no separate history: the normal request transaction supplies its
// committed three-token suffix. Hashing and selected-row copies run together,
// directly into the caller's already-retired pinned staging slot.
class NativeC1Prepare {
    using I64 = py::array_t<int64_t, py::array::c_style>;
    using U8 = py::array;
    I64 token_map_, multipliers_, primes_, offsets_;
    std::vector<std::shared_ptr<HostRows>> tables_;
    std::vector<std::vector<U8>> targets_;
    std::vector<uint64_t> slot_generations_;
    int64_t first_[2], last_[2], pad_;
    uint64_t generation_ = 0, history_generation_ = 0;
    std::string request_;
    bool pending_ = false, failed_ = false;
    std::mutex mutex_;
 public:
    NativeC1Prepare(I64 tokenMap, I64 multipliers, I64 primes, I64 offsets,
                    I64 first, I64 last, int64_t pad,
                    std::vector<std::shared_ptr<HostRows>> tables,
                    std::vector<std::vector<U8>> targets)
        : token_map_(std::move(tokenMap)), multipliers_(std::move(multipliers)),
          primes_(std::move(primes)), offsets_(std::move(offsets)), tables_(std::move(tables)),
          targets_(std::move(targets)), slot_generations_(targets_.size(), 0), pad_(pad) {
        require(token_map_.ndim() == 1 && token_map_.size() > 0 && pad_ >= 0,
                "Invalid C1 compressed token map");
        require(multipliers_.ndim() == 2 && multipliers_.shape(0) == 2 && multipliers_.shape(1) == 4 &&
                primes_.ndim() == 2 && primes_.shape(0) == 2 && primes_.shape(1) == 24 &&
                offsets_.ndim() == 2 && offsets_.shape(0) == 2 && offsets_.shape(1) == 24 &&
                first.size() == 2 && last.size() == 2 && tables_.size() == 2 && targets_.size() >= 2,
                "C1 Engram requires two layers, 2/3/4 grams and eight heads");
        for (size_t layer = 0; layer < 2; ++layer) {
            first_[layer] = first.data()[layer]; last_[layer] = last.data()[layer];
            require(first_[layer] >= 0 && first_[layer] < last_[layer] && last_[layer] <= 24 && tables_[layer],
                    "Invalid C1 TP head ownership");
            for (size_t head = 0; head < 24; ++head)
                require(primes_.data()[layer * 24 + head] > 0 && offsets_.data()[layer * 24 + head] >= 0,
                        "Invalid C1 hash divisor or offset");
        }
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        for (auto& slot : targets_) {
            require(slot.size() == 2, "C1 staging must bind both Engram layers");
            for (size_t layer = 0; layer < 2; ++layer) {
                auto& target = slot[layer];
                const auto width = tables_[layer]->width + tables_[layer]->groups;
                require(target.ndim() == 3 && target.shape(0) == 1 &&
                        target.shape(1) == last_[layer] - first_[layer] && target.shape(2) == width && target.writeable() &&
                        target.dtype().is(py::dtype::of<uint8_t>()) && (target.flags() & py::array::c_style),
                        "C1 pinned destination shape, layout or writeability differs");
                const auto begin = reinterpret_cast<uintptr_t>(target.data());
                const auto end = begin + target.nbytes();
                require(end >= begin, "C1 staging range overflow");
                for (const auto& range : ranges)
                    require(end <= range.first || range.second <= begin, "C1 staging slots alias");
                ranges.emplace_back(begin, end);
            }
        }
    }

    py::tuple prepare(const std::string& request, uint64_t historyGeneration, uint64_t generation,
                      size_t slot, int64_t token, bool image, I64 history, bool lateOnly) {
        // No Python operation occurs while the native mutex is held without
        // the GIL. Complete cannot race the transaction that creates its token.
        require(history.ndim() == 1 && history.size() <= 3, "C1 history suffix exceeds three tokens");
        require(token >= 0 && token < token_map_.size(), "C1 token is outside the frozen vocabulary");
        require(slot < targets_.size(), "C1 staging slot is out of range");
        py::array_t<int64_t> compressed({py::ssize_t(1)});
        py::array_t<int32_t> hashes({py::ssize_t(1), py::ssize_t(2), py::ssize_t(24)});
        const int64_t code = image ? -1 : token_map_.data()[token];
        require(code >= -1, "Invalid compressed C1 token");
        compressed.mutable_data()[0] = code;
        auto* hash = hashes.mutable_data();
        std::vector<uint8_t*> destinations;
        for (auto& target : targets_[slot]) destinations.push_back(static_cast<uint8_t*>(target.mutable_data()));
        long faults = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            require(!pending_ && !failed_, "C1 preparation has an unretired or failed generation");
            require(generation > generation_ && generation > slot_generations_[slot], "Stale C1 staging generation");
            try {
                uint64_t rolling[2] = {0, 0};
                bool blocked = false;
                for (size_t shift = 0; shift < 4; ++shift) {
                    int64_t value = code;
                    if (shift) {
                        if (shift > static_cast<size_t>(history.size())) { blocked = true; value = pad_; }
                        else value = history.data()[history.size() - shift];
                    }
                    blocked = blocked || value == -1;
                    if (blocked) value = pad_;
                    require(value >= 0, "Invalid committed C1 history code");
                    for (size_t layer = 0; layer < 2; ++layer) {
                        // Defined modulo-2^64 arithmetic matches NumPy int64
                        // multiplication/XOR, including signed-overflow cases.
                        rolling[layer] ^= uint64_t(value) * uint64_t(multipliers_.data()[layer * 4 + shift]);
                        if (!shift) continue;
                        int64_t signedBits;
                        std::memcpy(&signedBits, &rolling[layer], sizeof(signedBits));
                        for (size_t head = (shift - 1) * 8; head < shift * 8; ++head) {
                            const size_t index = layer * 24 + head;
                            const int64_t divisor = primes_.data()[index];
                            int64_t remainder = signedBits % divisor;
                            if (remainder < 0) remainder += divisor;
                            const int64_t offset = offsets_.data()[index];
                            require(offset <= INT32_MAX && remainder <= INT32_MAX - offset,
                                    "C1 hash row exceeds the int32 checkpoint contract");
                            hash[index] = int32_t(remainder + offset);
                        }
                    }
                }
                // Validate both layers before modifying either destination.
                const size_t firstLayer = lateOnly ? 1 : 0;
                for (size_t layer = firstLayer; layer < 2; ++layer)
                    for (int64_t head = first_[layer]; head < last_[layer]; ++head)
                        require(hash[layer * 24 + head] >= tables_[layer]->start &&
                                hash[layer * 24 + head] < tables_[layer]->stop, "C1 row does not belong to its TP shard");
                struct rusage before{}, after{};
                getrusage(RUSAGE_THREAD, &before);
                for (size_t layer = firstLayer; layer < 2; ++layer)
                    tables_[layer]->gather_packed(hash + layer * 24 + first_[layer],
                                                  last_[layer] - first_[layer], destinations[layer]);
                getrusage(RUSAGE_THREAD, &after);
                faults = after.ru_majflt - before.ru_majflt;
                request_ = request; history_generation_ = historyGeneration;
                generation_ = generation; slot_generations_[slot] = generation; pending_ = true;
            } catch (...) { failed_ = true; throw; }
        }
        return py::make_tuple(compressed, hashes, faults);
    }

    void complete(const std::string& request, uint64_t historyGeneration, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(pending_ && !failed_ && request == request_ && historyGeneration == history_generation_ &&
                generation == generation_, "Cannot retire a stale C1 transaction");
        pending_ = false;
    }
};
}

PYBIND11_MODULE(dsv41_host_gather, m) {
    m.attr("abi_version") = 1;
    m.attr("c1_abi_version") = 2;
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
    using I64 = py::array_t<int64_t, py::array::c_style>;
    using U8 = py::array;
    py::class_<NativeC1Prepare>(m, "NativeC1Prepare")
        .def(py::init<I64, I64, I64, I64, I64, I64, int64_t,
             std::vector<std::shared_ptr<HostRows>>, std::vector<std::vector<U8>>>())
        .def("prepare", &NativeC1Prepare::prepare,
             py::arg("request"), py::arg("history_generation"), py::arg("generation"), py::arg("slot"),
             py::arg("token"), py::arg("image"), py::arg("history"), py::arg("late_only") = false)
        .def("complete", &NativeC1Prepare::complete);
}
