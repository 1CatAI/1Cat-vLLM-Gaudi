// SPDX-License-Identifier: Apache-2.0
#pragma once

// Opt-in diagnostics. Markers follow the eager pipeline but introduce no
// stream waits; the production event synchronization retains both phases.
namespace dsv41_timing {
inline uint64_t monotonicNs() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return uint64_t(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

struct Marker {
  synEventHandle event = nullptr;
  uint64_t queued_ns = monotonicNs();
  uint64_t submit_begin_ns = 0, submit_end_ns = 0;
  std::atomic<bool> submitted{false};
  ~Marker() {
    if (event && habana::HPUDeviceContext::is_device_acquired())
      synEventDestroy(event);
  }
};

inline std::shared_ptr<Marker> record() {
  auto marker = std::make_shared<Marker>();
  auto stream_id = c10::hpu::getCurrentHPUStream().stream();
  habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
      [marker, stream_id]() {
        auto& device = habana::HPUDeviceContext::get_device();
        TORCH_CHECK(synEventCreate(&marker->event, device.id(), EVENT_COLLECT_TIME) == synSuccess,
                    "Verify diagnostic event allocation failed");
        marker->submit_begin_ns = monotonicNs();
        TORCH_CHECK(synEventRecord(marker->event, device.get_stream(stream_id)) == synSuccess,
                    "Verify diagnostic event recording failed");
        marker->submit_end_ns = monotonicNs();
        marker->submitted.store(true, std::memory_order_release);
      });
  return marker;
}

inline pybind11::dict snapshot(const std::shared_ptr<Marker>& marker, bool wait) {
  if (wait) {
    // Used only for clock calibration before serving, never between phases.
    pybind11::gil_scoped_release release;
    habana::eager::JoinPendingPipelineThreads();
    TORCH_CHECK(marker->submitted.load(std::memory_order_acquire), "Verify marker was not submitted");
    TORCH_CHECK(synEventSynchronize(marker->event) == synSuccess, "Verify marker wait failed");
  }
  pybind11::dict result;
  result["queued_ns"] = marker->queued_ns;
  if (!marker->submitted.load(std::memory_order_acquire)) {
    result["status"] = "not_submitted";
    return result;
  }
  result["submit_begin_ns"] = marker->submit_begin_ns;
  result["submit_end_ns"] = marker->submit_end_ns;
  uint64_t timestamp = 0;
  auto status = synEventElapsedTime(&timestamp, marker->event, nullptr);
  result["synapse_status"] = int(status);
  if (status == synSuccess) {
    result["status"] = "complete";
    result["device_timestamp_ns"] = timestamp;
  } else {
    result["status"] = "pending_or_unavailable";
  }
  return result;
}

inline pybind11::dict synchronize(uint64_t event_id) {
  uint64_t before, device_done, callbacks_done;
  {
    pybind11::gil_scoped_release release;
    auto& device = habana::HPUDeviceContext::get_device();
    before = monotonicNs();
    device.synchronize_event(event_id);
    device_done = monotonicNs();
    device.flush_host_events();
    callbacks_done = monotonicNs();
  }
  pybind11::dict result;
  result["event_wait_start_ns"] = before;
  result["event_wait_done_ns"] = device_done;
  result["host_callbacks_done_ns"] = callbacks_done;
  return result;
}

inline void bind(pybind11::module_& module) {
  pybind11::class_<Marker, std::shared_ptr<Marker>>(module, "VerifyPhaseMarker")
      .def("snapshot", &snapshot, pybind11::arg("wait") = false);
  module.def("verify_phase_marker", &record);
  module.def("verify_synchronize_phases", &synchronize);
}
} // namespace dsv41_timing
