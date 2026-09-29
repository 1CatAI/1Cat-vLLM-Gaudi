// SPDX-License-Identifier: Apache-2.0
// Diagnostic-only compute command replay. Every caller must synchronize before
// updating inputs, inspecting outputs or releasing fixed recipe allocations.
#include <torch/extension.h>
#include <dlfcn.h>
#include <habanalabs/synapse_api.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"
#include "habana_eager/eager_pipeline_utils.h"

namespace {
struct Sync { uint32_t index = 0; uint64_t target = 0; };
struct Info {
  int state = 0; uint64_t segments = 0; uint64_t replays = 0;
  uint32_t completion = 0; uint64_t target = 0;
  uint64_t global_bytes = 0; uint64_t arc_bytes = 0;
};
template<class T> T symbol(const char* name) {
  auto result = reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
  TORCH_CHECK(result, "Missing pinned native compute API: ", name);
  return result;
}
void check(synStatus status) { TORCH_CHECK(status == synSuccess, "Native compute diagnostic failed: ", status); }
class Compute : public std::enable_shared_from_this<Compute> {
 public:
  using Op = synStatus(*)(void*);
  using Create = synStatus(*)(void**, synStreamHandle);
  using Get = synStatus(*)(void*, Info*);
  using Segment = synStatus(*)(void*, uint64_t, const Sync*, uint8_t, Sync*);
  Compute() : create_(symbol<Create>("synNativeComputeGraphCreate")),
      begin_(symbol<Op>("synNativeComputeGraphBeginCapture")),
      end_(symbol<Op>("synNativeComputeGraphEndCapture")),
      start_(symbol<Op>("synNativeComputeGraphBeginReplay")),
      segment_(symbol<Segment>("synNativeComputeGraphReplaySegment")),
      get_(symbol<Get>("synNativeComputeGraphGetInfo")),
      destroy_(symbol<Op>("synNativeComputeGraphDestroy")) {}
  void begin() {
    habana::eager::JoinPendingPipelineThreads();
    auto self = shared_from_this();
    habana::HPUDeviceContext::execute_thread().enqueue([self]() {
      TORCH_CHECK(!self->graph_, "Repeated micro capture");
      self->stream_ = habana::HPUDeviceContext::get_device().get_stream(0);
      check(self->create_(&self->graph_, self->stream_)); check(self->begin_(self->graph_));
    });
    habana::eager::JoinPendingPipelineThreads();
  }
  void end(uint64_t segments) {
    habana::eager::JoinPendingPipelineThreads();
    auto self = shared_from_this();
    habana::HPUDeviceContext::execute_thread().enqueue([self, segments]() {
      check(self->end_(self->graph_)); check(self->get_(self->graph_, &self->info_));
      TORCH_CHECK(self->info_.state == 2 && self->info_.segments == segments,
                  "Native compute capture did not retain every recipe");
      self->ready_ = true;
    });
    habana::eager::JoinPendingPipelineThreads();
  }
  void replay() {
    TORCH_CHECK(ready_, "Native compute replay before capture");
    auto self = shared_from_this();
    // Keep event recording ordered with the normal Bridge lowering pipeline.
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([self]() {
      habana::HPUDeviceContext::execute_thread().enqueue([self]() {
        check(self->start_(self->graph_));
        Sync completion;
        for (uint64_t i = 0; i < self->info_.segments; ++i)
          check(self->segment_(self->graph_, i, nullptr, i + 1 == self->info_.segments, &completion));
      });
    });
  }
  static void replay_bounded_tiles(const std::vector<std::shared_ptr<Compute>>& chain, uint64_t tiles) {
    // Diagnostic capability gate: init/compact, eight fixed tile programs,
    // then the real consumer. All bindings and recipes are prepared once.
    // The bound comes from scheduler-owned lengths, never a device readback.
    TORCH_CHECK(chain.size() == 10 && tiles <= 8, "Expected init + eight tiles + consumer");
    for (const auto& part : chain)
      TORCH_CHECK(part && part->ready_ && part->stream_ == chain.front()->stream_,
                  "Bounded tile chain must retain ready programs on one stream");
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([chain, tiles]() {
      habana::HPUDeviceContext::execute_thread().enqueue([chain, tiles]() {
        for (size_t p = 0; p < chain.size(); ++p) {
          if (p > tiles && p < 9) continue;
          const auto& part = chain[p];
          check(part->start_(part->graph_));
          Sync completion;
          for (uint64_t s = 0; s < part->info_.segments; ++s)
            check(part->segment_(part->graph_, s, nullptr,
                                s + 1 == part->info_.segments, &completion));
        }
      });
    });
  }
  std::vector<uint64_t> info() {
    habana::eager::JoinPendingPipelineThreads();
    check(get_(graph_, &info_));
    return {info_.segments, info_.replays, info_.global_bytes, info_.arc_bytes};
  }
  uint64_t workspace_bytes() {
    habana::eager::JoinPendingPipelineThreads();
    TORCH_CHECK(ready_ && graph_, "Workspace query requires a retained compute graph");
    using GetWorkspace = synStatus(*)(void*, uint64_t*);
    uint64_t bytes = 0;
    check(symbol<GetWorkspace>("synNativeComputeGraphGetWorkspaceBytes")(graph_, &bytes));
    return bytes;
  }
  void close() {
    habana::eager::JoinPendingPipelineThreads();
    if (!graph_) return;
    check(synStreamSynchronize(stream_));
    check(destroy_(graph_)); graph_ = nullptr; ready_ = false;
  }
 private:
  Create create_; Op begin_, end_, start_; Segment segment_; Get get_; Op destroy_;
  void* graph_ = nullptr; synStreamHandle stream_ = nullptr; Info info_; bool ready_ = false;
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("replay_bounded_tiles", &Compute::replay_bounded_tiles);
  pybind11::class_<Compute, std::shared_ptr<Compute>>(m, "Compute")
    .def(pybind11::init<>()).def("begin", &Compute::begin).def("end", &Compute::end)
    .def("replay", &Compute::replay).def("info", &Compute::info)
    .def("workspace_bytes", &Compute::workspace_bytes).def("close", &Compute::close);
}
