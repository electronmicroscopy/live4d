#include "nxmx_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dectris::nxmx {
namespace {

namespace fs = std::filesystem;
using cbor::Value;

constexpr const char* kAngstrom = "angstrom";
constexpr const char* kDegree = "degree";
constexpr const char* kElectronVolt = "eV";
constexpr const char* kMeter = "m";
constexpr const char* kPixel = "pixel";
constexpr const char* kSecond = "s";
constexpr const char* kTempSuffix = ".tmp";
constexpr const char* kDataPath = "/entry/data/data";

template <typename T>
T check(T result, const char* what) {
  if (result < 0) throw std::runtime_error(std::string("HDF5 failed: ") + what);
  return result;
}

// Owns one HDF5 identifier of any kind.
class Id {
 public:
  explicit Id(hid_t id = -1) : id_(id) {}
  Id(hid_t id, const char* what) : id_(check(id, what)) {}
  Id(const Id&) = delete;
  Id& operator=(const Id&) = delete;
  Id(Id&& other) noexcept : id_(other.id_) { other.id_ = -1; }
  ~Id() { if (id_ >= 0) H5Idec_ref(id_); }
  operator hid_t() const { return id_; }
  hid_t release() { const hid_t id = id_; id_ = -1; return id; }

 private:
  hid_t id_;
};

Id file_access() {
  Id fapl(H5Pcreate(H5P_FILE_ACCESS), "create file access list");
  // The newest format available in HDF5 1.10, which Ubuntu ships; imagemux uses 1.14.
  check(H5Pset_libver_bounds(fapl, H5F_LIBVER_V110, H5F_LIBVER_V110), "set library bounds");
  return fapl;
}

Id utf8_string() {
  Id type(H5Tcopy(H5T_C_S1), "copy string type");
  check(H5Tset_size(type, H5T_VARIABLE), "set string size");
  check(H5Tset_cset(type, H5T_CSET_UTF8), "set string charset");
  return type;
}

Id bool_type() {
  Id type(H5Tenum_create(H5T_NATIVE_INT8), "create bool type");
  int8_t value = 0;
  check(H5Tenum_insert(type, "FALSE", &value), "insert FALSE");
  value = 1;
  check(H5Tenum_insert(type, "TRUE", &value), "insert TRUE");
  return type;
}

Id simple_space(std::initializer_list<hsize_t> dims) {
  std::vector<hsize_t> d(dims);
  return Id(H5Screate_simple(static_cast<int>(d.size()), d.data(), nullptr), "create dataspace");
}

Id scalar_space() { return Id(H5Screate(H5S_SCALAR), "create scalar dataspace"); }

Id group(hid_t parent, const char* name, const char* nx_class);

void attr(hid_t loc, const char* name, hid_t type, hid_t space, const void* value) {
  Id a(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), name);
  check(H5Awrite(a, type, value), name);
}

void attr(hid_t loc, const char* name, std::string_view value) {
  const std::string text(value);
  const char* p = text.c_str();
  attr(loc, name, utf8_string(), scalar_space(), &p);
}

void attr(hid_t loc, const char* name, const std::vector<std::string>& values) {
  std::vector<const char*> pointers;
  for (const auto& v : values) pointers.push_back(v.c_str());
  attr(loc, name, utf8_string(), simple_space({values.size()}), pointers.data());
}

void attr(hid_t loc, const char* name, const std::vector<double>& values) {
  attr(loc, name, H5T_NATIVE_DOUBLE, simple_space({values.size()}), values.data());
}

void attr_u8(hid_t loc, const char* name, uint8_t value) {
  attr(loc, name, H5T_NATIVE_UINT8, scalar_space(), &value);
}

void attr_fixed_ascii(hid_t loc, const char* name, std::string_view value) {
  Id type(H5Tcopy(H5T_C_S1), "copy string type");
  check(H5Tset_size(type, value.size()), "set string size");
  check(H5Tset_strpad(type, H5T_STR_NULLPAD), "set string padding");
  attr(loc, name, type, scalar_space(), value.data());
}

Id dataset(hid_t loc, const char* name, hid_t type, hid_t space, const void* value) {
  Id d(H5Dcreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name);
  check(H5Dwrite(d, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value), name);
  return d;
}

Id dataset(hid_t loc, const char* name, std::string_view value) {
  const std::string text(value);
  const char* p = text.c_str();
  return dataset(loc, name, utf8_string(), scalar_space(), &p);
}

Id dataset_f64(hid_t loc, const char* name, double value, const char* units = nullptr) {
  Id d = dataset(loc, name, H5T_NATIVE_DOUBLE, scalar_space(), &value);
  if (units) attr(d, "units", units);
  return d;
}

Id dataset_f64(hid_t loc, const char* name, const std::vector<double>& values,
               const char* units = nullptr) {
  Id d = dataset(loc, name, H5T_NATIVE_DOUBLE, simple_space({values.size()}), values.data());
  if (units) attr(d, "units", units);
  return d;
}

Id dataset_bool(hid_t loc, const char* name, bool value) {
  const int8_t v = value;
  return dataset(loc, name, bool_type(), scalar_space(), &v);
}

template <typename T>
Id dataset_scalar(hid_t loc, const char* name, hid_t type, T value) {
  return dataset(loc, name, type, scalar_space(), &value);
}

hid_t native_type(cbor::NumericArray::Type type) {
  using T = cbor::NumericArray::Type;
  switch (type) {
    case T::kUInt8: return H5T_NATIVE_UINT8;
    case T::kUInt16: return H5T_NATIVE_UINT16;
    case T::kUInt32: return H5T_NATIVE_UINT32;
    case T::kUInt64: return H5T_NATIVE_UINT64;
    case T::kInt8: return H5T_NATIVE_INT8;
    case T::kInt16: return H5T_NATIVE_INT16;
    case T::kInt32: return H5T_NATIVE_INT32;
    case T::kInt64: return H5T_NATIVE_INT64;
    case T::kFloat32: return H5T_NATIVE_FLOAT;
    default: return H5T_NATIVE_DOUBLE;
  }
}

void dataset_array(hid_t loc, const char* name, const cbor::NumericArray& array) {
  std::vector<hsize_t> dims(array.shape.begin(), array.shape.end());
  Id space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), name);
  dataset(loc, name, native_type(array.type), space, array.data.data());
}

Id group(hid_t parent, const char* name, const char* nx_class) {
  Id g(H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), name);
  attr(g, "NX_class", nx_class);
  return g;
}

Id open_or_create_group(hid_t parent, const std::string& name, const char* nx_class) {
  if (H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0)
    return Id(H5Gopen2(parent, name.c_str(), H5P_DEFAULT), name.c_str());
  return group(parent, name.c_str(), nx_class);
}

// Optional START fields written as datasets when present with the expected type.
void maybe_f64(hid_t loc, const char* name, const Value& start, const char* key,
               const char* units = nullptr) {
  if (const Value* v = start.get(key); v && v->as_double()) dataset_f64(loc, name, *v->as_double(), units);
}

void maybe_bool(hid_t loc, const char* name, const Value& start, const char* key) {
  if (const Value* v = start.get(key); v && v->as_bool()) dataset_bool(loc, name, *v->as_bool());
}

void maybe_text(hid_t loc, const char* name, const Value& start, const char* key) {
  if (const Value* v = start.get(key); v && v->as_text()) dataset(loc, name, *v->as_text());
}

std::string local_time_zone() {
  const std::time_t epoch = 0;
  std::tm local{};
  localtime_r(&epoch, &local);
  const long offset = local.tm_gmtoff;
  const long minutes = (offset < 0 ? -offset : offset) / 60;
  char text[32];
  std::snprintf(text, sizeof(text), "%c%02ld:%02ld", offset < 0 ? '-' : '+', minutes / 60,
                minutes % 60);
  return text;
}

struct Rotation {
  double start{};
  double increment{};
};

std::optional<Rotation> rotation(const Value& start, const char* axis) {
  const Value* goniometer = start.get("goniometer");
  const Value* v = goniometer ? goniometer->get(axis) : nullptr;
  if (!v) return std::nullopt;
  const Value* s = v->get("start");
  const Value* i = v->get("increment");
  if (!s || !i || !s->as_double() || !i->as_double()) return std::nullopt;
  return Rotation{*s->as_double(), *i->as_double()};
}

// Axis positions per image with imagemux's NXmx 2024 fields: <axis>, <axis>_end and
// <axis>_increment_set, in degrees.
void write_axis(hid_t loc, const char* name, const Rotation& r, uint64_t images) {
  std::vector<double> angles(images + 1);
  for (uint64_t n = 0; n <= images; ++n) angles[n] = r.start + r.increment * static_cast<double>(n);
  Id axis = dataset_f64(loc, name, std::vector<double>(angles.begin(), angles.end() - 1), kDegree);
  attr(axis, "transformation_type", "rotation");
  const std::string end = std::string(name) + "_end";
  dataset_f64(loc, end.c_str(), std::vector<double>(angles.begin() + 1, angles.end()), kDegree);
  const std::string increment = std::string(name) + "_increment_set";
  dataset_f64(loc, increment.c_str(), r.increment, kDegree);
}

void write_entry(hid_t file, const SeriesLayout& layout) {
  attr(file, "default", "entry");
  Id entry = group(file, "entry", "NXentry");
  attr(entry, "default", "data");
  dataset(entry, "definition", "NXmx");
  attr(entry, "version", "1.0");
  attr_fixed_ascii(entry, "DECTRIS_lab_geometry", "mcstas");
  dataset(entry, "program_name", "live4d");

  Id data = group(entry, "data", "NXdata");
  attr(data, "signal", "data");
  attr(data, "axes", std::vector<std::string>{"image_id", "channel", ".", "."});
  attr(data, "default_slice", std::vector<std::string>{".", layout.channel, ".", "."});
  attr_u8(data, "image_id_indices", 0);
  attr_u8(data, "channel_indices", 1);

  const uint64_t n = layout.number_of_images;
  std::vector<uint64_t> ids(n);
  for (uint64_t i = 0; i < n; ++i) ids[i] = i;
  dataset(data, "image_id", H5T_NATIVE_UINT64, simple_space({n}), ids.data());
  const char* channel = layout.channel.c_str();
  dataset(data, "channel", utf8_string(), simple_space({1}), &channel);

  Id virtual_space = simple_space({n, 1, layout.height, layout.width});
  Id dcpl(H5Pcreate(H5P_DATASET_CREATE), "create VDS properties");
  check(H5Pset_fill_value(dcpl, H5T_NATIVE_UINT16, &kNoData), "set VDS fill value");
  for (uint64_t file_number = 1; file_number <= layout.data_files(); ++file_number) {
    const hsize_t count = layout.images_in_file(file_number);
    const hsize_t start[4] = {layout.first_image(file_number), 0, 0, 0};
    const hsize_t block[4] = {count, 1, layout.height, layout.width};
    Id source = simple_space({count, 1, layout.height, layout.width});
    check(H5Sselect_hyperslab(virtual_space, H5S_SELECT_SET, start, nullptr, block, nullptr),
          "select VDS block");
    const std::string name = layout.data_file(file_number).filename().string();
    check(H5Pset_virtual(dcpl, virtual_space, name.c_str(), kDataPath, source), "map VDS block");
  }
  check(H5Sselect_all(virtual_space), "select VDS");
  Id vds(H5Dcreate2(data, "data", H5T_STD_U16LE, virtual_space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
         "create VDS");
}

void write_detector(hid_t file, const SeriesLayout& layout, const Value& start) {
  Id instrument = group(file, "/entry/instrument", "NXinstrument");
  dataset(instrument, "time_zone", local_time_zone());

  Id beam = group(instrument, "beam", "NXbeam");
  maybe_f64(beam, "incident_wavelength", start, "incident_wavelength", kAngstrom);
  maybe_f64(beam, "incident_energy", start, "incident_energy", kElectronVolt);

  const auto two_theta = rotation(start, "two_theta");
  std::optional<std::vector<double>> translation;
  if (const Value* t = start.get("detector_translation")) {
    const auto array = cbor::numeric_array(t->untagged());
    if (array && array->type == cbor::NumericArray::Type::kFloat64 && array->elements() == 3) {
      translation.emplace(3);
      std::memcpy(translation->data(), array->data.data(), 3 * sizeof(double));
    }
  }
  double translation_length = 0;
  if (translation) {
    for (double x : *translation) translation_length += x * x;
    translation_length = std::sqrt(translation_length);
    if (translation_length <= 0) translation.reset();
  }

  Id detector = group(instrument, "detector", "NXdetector");
  maybe_f64(detector, "beam_center_x", start, "beam_center_x", kPixel);
  maybe_f64(detector, "beam_center_y", start, "beam_center_y", kPixel);
  if (const Value* v = start.get("bit_depth_readout"); v && v->as_int())
    dataset_scalar<uint16_t>(detector, "bit_depth_readout", H5T_NATIVE_UINT16,
                             static_cast<uint16_t>(*v->as_int()));
  maybe_f64(detector, "count_time", start, "count_time", kSecond);
  maybe_bool(detector, "countrate_correction_applied", start, "countrate_correction_enabled");
  if (const Value* v = start.get("countrate_correction_lookup_table"))
    if (const auto array = cbor::numeric_array(*v))
      dataset_array(detector, "countrate_correction_lookup_table", *array);
  if (translation) dataset(detector, "depends_on", "/entry/instrument/detector/geometry/translation");
  else if (two_theta) dataset(detector, "depends_on", "/entry/instrument/detector/transformations/two_theta");
  maybe_text(detector, "description", start, "detector_description");
  maybe_text(detector, "serial_number", start, "detector_serial_number");
  maybe_bool(detector, "flatfield_applied", start, "flatfield_enabled");
  maybe_f64(detector, "frame_time", start, "frame_time", kSecond);
  maybe_bool(detector, "pixel_mask_applied", start, "pixel_mask_enabled");
  if (const Value* v = start.get("saturation_value"); v && v->as_int())
    dataset_scalar<int64_t>(detector, "saturation_value", H5T_NATIVE_INT64, *v->as_int());
  maybe_text(detector, "sensor_material", start, "sensor_material");
  maybe_f64(detector, "sensor_thickness", start, "sensor_thickness", kMeter);
  dataset(detector, "type", "HPC");
  maybe_bool(detector, "virtual_pixel_interpolation_applied", start,
             "virtual_pixel_interpolation_enabled");
  maybe_f64(detector, "x_pixel_size", start, "pixel_size_x", kMeter);
  maybe_f64(detector, "y_pixel_size", start, "pixel_size_y", kMeter);

  // Per-channel calibration, keyed by channel name in START.
  const auto per_channel = [&](const char* key, auto&& write) {
    const Value* map = start.get(key);
    if (!map || map->untagged().kind != Value::Kind::kMap) return;
    const auto& items = map->untagged().items;
    for (size_t i = 0; i + 1 < items.size(); i += 2) {
      const auto name = items[i].as_text();
      if (!name) continue;
      Id channel = open_or_create_group(detector, *name + "_channel", "NXdetector_channel");
      write(channel, items[i + 1]);
    }
  };
  per_channel("threshold_energy", [](hid_t channel, const Value& v) {
    if (v.as_double()) dataset_f64(channel, "threshold_energy", *v.as_double(), kElectronVolt);
  });
  per_channel("flatfield", [](hid_t channel, const Value& v) {
    if (const auto array = cbor::numeric_array(v)) dataset_array(channel, "flatfield", *array);
  });
  per_channel("pixel_mask", [](hid_t channel, const Value& v) {
    if (const auto array = cbor::numeric_array(v)) dataset_array(channel, "pixel_mask", *array);
  });

  Id geometry = group(detector, "geometry", "NXtransformations");
  if (translation) {
    Id t = dataset_f64(geometry, "translation", translation_length);
    attr(t, "transformation_type", "translation");
    attr(t, "units", kMeter);
    attr(t, "vector", std::vector<double>{(*translation)[0] / translation_length,
                                          (*translation)[1] / translation_length,
                                          (*translation)[2] / translation_length});
    attr(t, "depends_on", two_theta ? "/entry/instrument/detector/transformations/two_theta" : ".");
  }

  Id module = group(detector, "module", "NXdetector_module");
  const int32_t origin[2] = {0, 0};
  const int32_t size[2] = {static_cast<int32_t>(layout.height), static_cast<int32_t>(layout.width)};
  dataset(module, "data_origin", H5T_NATIVE_INT32, simple_space({2}), origin);
  dataset(module, "data_size", H5T_NATIVE_INT32, simple_space({2}), size);
  for (const auto& [name, key] : {std::pair{"fast_pixel_direction", "pixel_size_x"},
                                  std::pair{"slow_pixel_direction", "pixel_size_y"}}) {
    const Value* v = start.get(key);
    if (!v || !v->as_double()) continue;
    Id direction = dataset_f64(module, name, *v->as_double(), kMeter);
    attr(direction, "transformation_type", "translation");
    attr(direction, "depends_on", ".");
    attr(direction, "offset", std::vector<double>{0.0, 0.0, 0.0});
  }

  Id transformations = group(detector, "transformations", "NXtransformations");
  if (two_theta) {
    write_axis(transformations, "two_theta", *two_theta, layout.number_of_images);
    Id axis(H5Dopen2(transformations, "two_theta", H5P_DEFAULT), "open two_theta");
    attr(axis, "depends_on", ".");
  }
}

void write_sample(hid_t file, const SeriesLayout& layout, const Value& start) {
  Id sample = group(file, "/entry/sample", "NXsample");
  dataset(sample, "depends_on", ".");
  Id transformations = group(sample, "transformations", "NXtransformations");
  for (const char* axis : {"omega", "chi", "kappa", "phi"})
    if (const auto r = rotation(start, axis)) write_axis(transformations, axis, *r, layout.number_of_images);
  Id source = group(file, "/entry/source", "NXsource");
}

std::string with_suffix(const fs::path& path, const char* suffix) {
  return path.string() + suffix;
}

}  // namespace

void register_filters() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (H5Zfilter_avail(kBitshuffleFilter) > 0) return;
    static const H5Z_class2_t placeholder{
        H5Z_CLASS_T_VERS, kBitshuffleFilter, 1, 1,
        "bitshuffle; see https://github.com/kiyo-masui/bitshuffle", nullptr, nullptr,
        [](unsigned, size_t, const unsigned[], size_t, size_t*, void**) -> size_t { return 0; }};
    check(H5Zregister(&placeholder), "register bitshuffle placeholder");
  });
}

uint64_t SeriesLayout::data_files() const {
  return (number_of_images + images_per_file - 1) / images_per_file;
}

uint64_t SeriesLayout::first_image(uint64_t file_number) const {
  return (file_number - 1) * images_per_file;
}

uint64_t SeriesLayout::images_in_file(uint64_t file_number) const {
  return std::min(images_per_file, number_of_images - first_image(file_number));
}

fs::path SeriesLayout::master_file() const { return with_suffix(prefix, "_master.h5"); }

fs::path SeriesLayout::data_file(uint64_t file_number) const {
  char suffix[32];
  std::snprintf(suffix, sizeof(suffix), "_data_%06llu.h5",
                static_cast<unsigned long long>(file_number));
  return with_suffix(prefix, suffix);
}

void write_master_file(const SeriesLayout& layout, const Value& start) {
  if (layout.number_of_images == 0 || layout.images_per_file == 0)
    throw std::invalid_argument("NXmx series requires images and images per file");
  const fs::path path = layout.master_file();
  const fs::path temp = with_suffix(path, kTempSuffix);
  fs::create_directories(path.parent_path());
  {
    Id file(H5Fcreate(temp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, file_access()), "create master file");
    try {
      write_entry(file, layout);
      write_detector(file, layout, start);
      write_sample(file, layout, start);
      check(H5Fflush(file, H5F_SCOPE_LOCAL), "flush master file");
    } catch (...) {
      std::error_code ignored;
      fs::remove(temp, ignored);
      throw;
    }
  }
  fs::rename(temp, path);
}

DataFile::~DataFile() {
  try {
    close();
  } catch (...) {
  }
}

void DataFile::open(const SeriesLayout& layout, uint64_t file_number) {
  close();
  path_ = layout.data_file(file_number);
  file_number_ = file_number;
  first_image_ = layout.first_image(file_number);
  images_ = layout.images_in_file(file_number);
  written_ = 0;
  if (fs::exists(path_)) {
    // Images that arrive after their file was completed are added to it in place.
    temp_path_.clear();
    Id file(H5Fopen(path_.c_str(), H5F_ACC_RDWR, file_access()), "reopen data file");
    dataset_ = check(H5Dopen2(file, kDataPath, H5P_DEFAULT), "open data file dataset");
    file_ = file.release();
    return;
  }
  temp_path_ = with_suffix(path_, kTempSuffix);
  Id file(H5Fcreate(temp_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, file_access()), "create data file");
  Id entry = group(file, "entry", "NXentry");
  Id data = group(entry, "data", "NXdata");
  const hsize_t chunk[4] = {1, 1, layout.height, layout.width};
  Id dcpl(H5Pcreate(H5P_DATASET_CREATE), "create dataset properties");
  check(H5Pset_chunk(dcpl, 4, chunk), "set chunk shape");
  check(H5Pset_fill_value(dcpl, H5T_NATIVE_UINT16, &kNoData), "set fill value");
  // Bitshuffle options: version, element size, default block size and zstd, as imagemux.
  const unsigned options[5] = {kBitshuffleZstdVersion[0], kBitshuffleZstdVersion[1],
                               sizeof(uint16_t), 0, kBitshuffleCompressZstd};
  check(H5Pset_filter(dcpl, kBitshuffleFilter, H5Z_FLAG_MANDATORY, 5, options), "set bitshuffle filter");
  Id space = simple_space({images_, 1, layout.height, layout.width});
  dataset_ = check(H5Dcreate2(data, "data", H5T_STD_U16LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
                   "create data file dataset");
  file_ = file.release();
}

void DataFile::write_chunk(uint64_t image_id, const void* data, size_t bytes) {
  if (image_id < first_image_ || image_id - first_image_ >= images_)
    throw std::out_of_range("image is outside the open data file");
  hsize_t offset[4] = {image_id - first_image_, 0, 0, 0};
  check(H5Dwrite_chunk(dataset_, H5P_DEFAULT, 0, offset, bytes, data), "write image chunk");
  ++written_;
}

void DataFile::close() {
  if (file_ < 0) return;
  const hid_t dataset = dataset_, file = file_;
  dataset_ = file_ = -1;
  const herr_t dataset_status = H5Dclose(dataset);
  check(H5Fclose(file), "close data file");
  check(dataset_status, "close data file dataset");
  if (!temp_path_.empty()) fs::rename(temp_path_, path_);
  temp_path_.clear();
}

}  // namespace dectris::nxmx
