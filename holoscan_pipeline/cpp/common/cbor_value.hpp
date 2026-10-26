#pragma once

// Minimal CBOR (RFC 8949) decoder for series metadata. START is decoded once per series,
// outside the image path; images use fast_cbor_parse instead.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dectris::cbor {

struct Value {
  enum class Kind { kNull, kBool, kUnsigned, kNegative, kFloat, kText, kBytes, kArray, kMap, kTag };
  Kind kind{Kind::kNull};
  bool boolean{};
  uint64_t integer{};  // kUnsigned value, kNegative encoded as -1 - integer, or the kTag number
  double real{};
  std::string bytes;          // kText or kBytes contents
  std::vector<Value> items;   // kArray elements, kMap alternating key/value, kTag content
  // Value of a text key in a map (tags on the map are skipped); nullptr when absent.
  const Value* get(std::string_view key) const;
  const Value& untagged() const;
  bool is_text() const { return untagged().kind == Kind::kText; }
  std::optional<double> as_double() const;
  std::optional<int64_t> as_int() const;
  std::optional<bool> as_bool() const;
  std::optional<std::string> as_text() const;
};

// Throws std::runtime_error for malformed or overly nested input.
Value decode(const uint8_t* data, size_t size);

// Numeric arrays: RFC 8746 typed arrays, optionally inside a tag-40 row-major
// multi-dimensional array, or plain CBOR arrays of numbers.
struct NumericArray {
  enum class Type { kUInt8, kUInt16, kUInt32, kUInt64, kInt8, kInt16, kInt32, kInt64,
                    kFloat32, kFloat64 };
  Type type{Type::kFloat64};
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;  // native little-endian elements, row-major
  size_t element_size() const;
  size_t elements() const;
};

// Returns nothing for values that are not uncompressed numeric arrays.
std::optional<NumericArray> numeric_array(const Value& value);

}  // namespace dectris::cbor
