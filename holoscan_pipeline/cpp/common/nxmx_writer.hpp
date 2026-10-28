#pragma once

// NeXus NXmx (2024) HDF5 layout following imagemux's NexusV20242Nxmx filewriter: a master
// file with START metadata and a virtual /entry/data/data over data files
// <prefix>_data_NNNNNN.h5 that each hold up to images_per_file images. Image chunks
// arrive pre-compressed in bitshuffle/zstd HDF5 framing and are stored without filtering.
// With em_metadata_link (imagemux's DectrisEm format), /entry/instrument/electron_microscope
// is an external link to /electron_microscope in <prefix>_em_metadata.h5.
// Host-only; every call holds one process-wide lock, so several writers may share a process.

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "cbor_value.hpp"

namespace dectris::nxmx {

// HDF5 filter identifiers and options matching imagemux and the bitshuffle HDF5 filter.
constexpr H5Z_filter_t kBitshuffleFilter = 32008;
constexpr unsigned kBitshuffleZstdVersion[2] = {0, 5};
constexpr unsigned kBitshuffleCompressZstd = 3;
constexpr uint16_t kNoData = UINT16_MAX;

// Register a placeholder bitshuffle filter so chunked datasets can declare it. Chunks are
// written with H5Dwrite_chunk, so the placeholder never runs; readers use the real filter.
void register_filters();

struct SeriesLayout {
  std::filesystem::path prefix;  // directory and file name stem, e.g. /data/series_7
  uint64_t number_of_images{};
  uint64_t images_per_file{};
  uint32_t height{};
  uint32_t width{};
  std::string channel{"threshold_1"};
  bool em_metadata_link{};

  uint64_t data_files() const;
  uint64_t first_image(uint64_t file_number) const;  // file_number counts from 1
  uint64_t images_in_file(uint64_t file_number) const;
  std::filesystem::path master_file() const;
  std::filesystem::path data_file(uint64_t file_number) const;
  std::filesystem::path em_metadata_file() const;
};

// Write <prefix>_master.h5 from the START map. The file appears under its final name
// only when complete.
void write_master_file(const SeriesLayout& layout, const cbor::Value& start);

// Write <prefix>_em_metadata.h5 in novena's EM metadata layout (format 1.0) from the START
// header_appendix electron_microscope map. Every dataset is written; absent ones get novena's
// defaults so novena's reader opens the file. Throws std::invalid_argument for a value of the
// wrong type and returns the keys that are not part of the format, which are not written.
std::vector<std::string> write_em_metadata_file(const SeriesLayout& layout,
                                                const cbor::Value& metadata);

class DataFile {
 public:
  DataFile() = default;
  DataFile(const DataFile&) = delete;
  DataFile& operator=(const DataFile&) = delete;
  ~DataFile();

  // Create data file file_number, or reopen it when it was already completed.
  void open(const SeriesLayout& layout, uint64_t file_number);
  bool is_open() const { return file_ >= 0; }
  uint64_t file_number() const { return file_number_; }
  uint64_t first_image() const { return first_image_; }
  uint64_t images() const { return images_; }
  uint64_t written() const { return written_; }
  // Store one framed bitshuffle/zstd chunk for image index image_id - first_image().
  void write_chunk(uint64_t image_id, const void* data, size_t bytes);
  // Close the file and make it visible under its final name.
  void close();

 private:
  hid_t file_{-1};
  hid_t dataset_{-1};
  std::filesystem::path path_;
  std::filesystem::path temp_path_;
  uint64_t file_number_{};
  uint64_t first_image_{};
  uint64_t images_{};
  uint64_t written_{};
};

}  // namespace dectris::nxmx
