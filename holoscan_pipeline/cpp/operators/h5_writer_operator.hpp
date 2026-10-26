#pragma once

#include <climits>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>

#include "batch_window.hpp"
#include "pipeline_types.hpp"

namespace holoscan::ops {

struct H5WriterParams {
  std::string output_dir{"."};
  std::string name_pattern{"series_$id"};
  uint64_t images_per_file{1000000};
  int64_t batch_size{512};
  int64_t staging_batches{4};
  int64_t device{};
  uint64_t publication_window{};
  uint64_t window_consumer{1};
  uint64_t completion_latch{};
};

HOLOSCAN_PARAM_SCHEMA(
    H5WriterParams, holoscan::param_schema("holoscan.dectris.H5WriterParams", 1),
    holoscan::field<&H5WriterParams::output_dir>("output_dir"),
    holoscan::field<&H5WriterParams::name_pattern>("name_pattern"),
    holoscan::field<&H5WriterParams::images_per_file>("images_per_file"),
    holoscan::field<&H5WriterParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&H5WriterParams::staging_batches>("staging_batches").range(int64_t{2}, int64_t{64}),
    holoscan::field<&H5WriterParams::device>("device").range(int64_t{0}, INT64_MAX),
    holoscan::field<&H5WriterParams::publication_window>("publication_window"),
    holoscan::field<&H5WriterParams::window_consumer>("window_consumer"),
    holoscan::field<&H5WriterParams::completion_latch>("completion_latch"));

// Writes each series as NXmx HDF5: a master file from START and data files of
// images_per_file images whose chunks are bitshuffle/zstd compressed on the GPU.
// The callback enqueues GPU compression into mapped pinned staging; one writer thread owns
// all HDF5 calls. Staging is bounded; a full staging ring delays the publication window.
class H5WriterOp final : public Operator<H5WriterParams> {
 public:
  ~H5WriterOp() override;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  Input<DecodedBatch> input;

 private:
  // nvCOMP chunk arrays for one image geometry, laid out image-major over the batch.
  struct Geometry {
    uint32_t height{};
    uint32_t width{};
    size_t chunks{};  // bitshuffle blocks per image
    const void** in_ptrs{};
    size_t* in_sizes{};
    void** out_ptrs{};
  };
  struct Slot {
    uint8_t* host{};       // framed chunks, one stride per image, in mapped pinned memory
    uint8_t* device{};     // device alias of host
    uint32_t* sizes{};     // framed bytes per image, mapped; 0 marks a failed image
    uint32_t* device_sizes{};
    cudaEvent_t ready{};
    std::vector<uint64_t> ids;
  };
  enum class JobKind { kStart, kBatch, kEnd, kStop };
  struct Job {
    JobKind kind{};
    uint64_t series_id{};
    uint32_t height{};
    uint32_t width{};
    size_t slot{};
    uint64_t series_images{};
    bool write{true};  // false when START header_appendix sets enable_h5_writer to false
    std::vector<uint8_t> start_metadata;
  };

  void allocate();
  void release_resources();
  Geometry& geometry_for(uint32_t height, uint32_t width);
  void compress(const uint16_t* frames, size_t count, const Geometry& geometry, Slot& slot,
                cudaStream_t stream);
  void push(Job job);
  void writer_loop();
  std::string error() const;

  std::vector<Geometry> geometries_;
  std::vector<Slot> slots_;
  uint8_t* d_shuffled_{};
  uint8_t* d_compressed_{};
  size_t* d_compressed_sizes_{};
  nvcompStatus_t* d_statuses_{};
  void* d_temp_{};
  size_t temp_bytes_{};
  size_t chunk_stride_{};  // nvCOMP output bytes reserved per chunk
  size_t slot_stride_{};   // framed bytes reserved per image
  size_t batch_capacity_{};
  cudaEvent_t last_ready_{};
  ::dectris::BatchWindow* window_{};
  ::dectris::CompletionLatch* completion_{};

  bool active_series_{};
  bool active_writing_{};
  uint64_t active_series_id_{};
  uint32_t active_height_{};
  uint32_t active_width_{};

  mutable std::mutex mutex_;
  std::condition_variable jobs_changed_;
  std::condition_variable slots_changed_;
  std::deque<Job> jobs_;
  std::deque<size_t> free_slots_;
  uint64_t ends_pushed_{};  // callback thread only
  uint64_t ends_done_{};
  std::string error_;
  std::thread writer_;
};

}  // namespace holoscan::ops
