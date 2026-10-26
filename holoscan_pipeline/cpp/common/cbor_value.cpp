#include "cbor_value.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace dectris::cbor {
namespace {

constexpr int kMaxDepth = 32;
constexpr uint64_t kSelfDescribedTag = 55799;
constexpr uint64_t kMultiDimTag = 40;

class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  bool done() const { return pos_ == size_; }

  Value value(int depth) {
    if (depth > kMaxDepth) throw std::runtime_error("CBOR nesting too deep");
    const uint8_t initial = byte();
    const uint8_t major = initial >> 5;
    const uint8_t info = initial & 31;
    Value v;
    if (major == 7) return simple(info);
    if (info == 31) return indefinite(major, depth);
    const uint64_t arg = argument(info);
    switch (major) {
      case 0: v.kind = Value::Kind::kUnsigned; v.integer = arg; break;
      case 1: v.kind = Value::Kind::kNegative; v.integer = arg; break;
      case 2: case 3:
        v.kind = major == 2 ? Value::Kind::kBytes : Value::Kind::kText;
        v.bytes.assign(reinterpret_cast<const char*>(take(arg)), arg);
        break;
      case 4: case 5: {
        v.kind = major == 4 ? Value::Kind::kArray : Value::Kind::kMap;
        const uint64_t count = major == 4 ? arg : arg * 2;
        if (count > size_ - pos_) throw std::runtime_error("CBOR container exceeds input");
        v.items.reserve(count);
        for (uint64_t i = 0; i < count; ++i) v.items.push_back(value(depth + 1));
        break;
      }
      case 6:
        v.kind = Value::Kind::kTag;
        v.integer = arg;
        v.items.push_back(value(depth + 1));
        break;
    }
    return v;
  }

 private:
  uint8_t byte() {
    if (pos_ >= size_) throw std::runtime_error("truncated CBOR");
    return data_[pos_++];
  }

  const uint8_t* take(uint64_t n) {
    if (n > size_ - pos_) throw std::runtime_error("truncated CBOR");
    const uint8_t* p = data_ + pos_;
    pos_ += n;
    return p;
  }

  uint64_t big_endian(int n) {
    const uint8_t* p = take(n);
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v = (v << 8) | p[i];
    return v;
  }

  uint64_t argument(uint8_t info) {
    if (info < 24) return info;
    if (info <= 27) return big_endian(1 << (info - 24));
    throw std::runtime_error("invalid CBOR argument");
  }

  Value simple(uint8_t info) {
    Value v;
    switch (info) {
      case 20: case 21: v.kind = Value::Kind::kBool; v.boolean = info == 21; return v;
      case 22: case 23: return v;  // null, undefined
      case 25: v.kind = Value::Kind::kFloat; v.real = half(big_endian(2)); return v;
      case 26: {
        v.kind = Value::Kind::kFloat;
        v.real = std::bit_cast<float>(static_cast<uint32_t>(big_endian(4)));
        return v;
      }
      case 27: v.kind = Value::Kind::kFloat; v.real = std::bit_cast<double>(big_endian(8)); return v;
      default:
        if (info < 24) return v;  // unassigned simple values decode as null
        throw std::runtime_error("unsupported CBOR simple value");
    }
  }

  static double half(uint64_t bits) {
    const int exponent = (bits >> 10) & 31;
    const double mantissa = static_cast<double>(bits & 1023);
    double value;
    if (exponent == 0) value = std::ldexp(mantissa, -24);
    else if (exponent == 31) value = mantissa ? std::numeric_limits<double>::quiet_NaN()
                                              : std::numeric_limits<double>::infinity();
    else value = std::ldexp(mantissa + 1024, exponent - 25);
    return (bits & 0x8000) ? -value : value;
  }

  Value indefinite(uint8_t major, int depth) {
    Value v;
    if (major == 2 || major == 3) {
      v.kind = major == 2 ? Value::Kind::kBytes : Value::Kind::kText;
      while (peek_break() == false) {
        const uint8_t initial = byte();
        if ((initial >> 5) != major || (initial & 31) == 31)
          throw std::runtime_error("invalid CBOR string chunk");
        const uint64_t n = argument(initial & 31);
        v.bytes.append(reinterpret_cast<const char*>(take(n)), n);
      }
      return v;
    }
    if (major != 4 && major != 5) throw std::runtime_error("invalid indefinite CBOR item");
    v.kind = major == 4 ? Value::Kind::kArray : Value::Kind::kMap;
    while (peek_break() == false) {
      v.items.push_back(value(depth + 1));
      if (major == 5) v.items.push_back(value(depth + 1));
    }
    return v;
  }

  bool peek_break() {
    if (pos_ >= size_) throw std::runtime_error("truncated CBOR");
    if (data_[pos_] != 0xff) return false;
    ++pos_;
    return true;
  }

  const uint8_t* data_;
  size_t size_;
  size_t pos_{};
};

size_t type_size(NumericArray::Type type) {
  switch (type) {
    case NumericArray::Type::kUInt8: case NumericArray::Type::kInt8: return 1;
    case NumericArray::Type::kUInt16: case NumericArray::Type::kInt16: return 2;
    case NumericArray::Type::kUInt32: case NumericArray::Type::kInt32:
    case NumericArray::Type::kFloat32: return 4;
    default: return 8;
  }
}

// RFC 8746 typed array tag 64..87 except clamped/half/quad: tag = 64 + 16f + 8s + 4e + ll.
std::optional<NumericArray> typed_array(uint64_t tag, const std::string& bytes) {
  if (tag < 64 || tag > 87 || tag == 68) return std::nullopt;
  const unsigned bits = static_cast<unsigned>(tag - 64);
  const bool is_float = bits & 16, is_signed = bits & 8, little = bits & 4;
  const unsigned ll = bits & 3;
  NumericArray array;
  using T = NumericArray::Type;
  if (is_float) {
    if (is_signed || ll == 0 || ll == 3) return std::nullopt;  // half and quad unsupported
    array.type = ll == 1 ? T::kFloat32 : T::kFloat64;
  } else {
    static constexpr T kUnsigned[] = {T::kUInt8, T::kUInt16, T::kUInt32, T::kUInt64};
    static constexpr T kSigned[] = {T::kInt8, T::kInt16, T::kInt32, T::kInt64};
    array.type = is_signed ? kSigned[ll] : kUnsigned[ll];
  }
  const size_t size = type_size(array.type);
  if (bytes.size() % size) return std::nullopt;
  array.data.assign(bytes.begin(), bytes.end());
  if (!little && size > 1) {
    for (size_t i = 0; i < array.data.size(); i += size)
      std::reverse(array.data.begin() + i, array.data.begin() + i + size);
  }
  array.shape = {bytes.size() / size};
  return array;
}

}  // namespace

const Value& Value::untagged() const {
  const Value* v = this;
  while (v->kind == Kind::kTag && !v->items.empty()) v = &v->items[0];
  return *v;
}

const Value* Value::get(std::string_view key) const {
  const Value& map = untagged();
  if (map.kind != Kind::kMap) return nullptr;
  for (size_t i = 0; i + 1 < map.items.size(); i += 2) {
    const Value& k = map.items[i].untagged();
    if (k.kind == Kind::kText && k.bytes == key) return &map.items[i + 1];
  }
  return nullptr;
}

std::optional<double> Value::as_double() const {
  const Value& v = untagged();
  switch (v.kind) {
    case Kind::kFloat: return v.real;
    case Kind::kUnsigned: return static_cast<double>(v.integer);
    case Kind::kNegative: return -1.0 - static_cast<double>(v.integer);
    default: return std::nullopt;
  }
}

std::optional<int64_t> Value::as_int() const {
  const Value& v = untagged();
  if (v.kind == Kind::kUnsigned && v.integer <= static_cast<uint64_t>(INT64_MAX))
    return static_cast<int64_t>(v.integer);
  if (v.kind == Kind::kNegative && v.integer <= static_cast<uint64_t>(INT64_MAX))
    return -1 - static_cast<int64_t>(v.integer);
  return std::nullopt;
}

std::optional<bool> Value::as_bool() const {
  const Value& v = untagged();
  if (v.kind == Kind::kBool) return v.boolean;
  return std::nullopt;
}

std::optional<std::string> Value::as_text() const {
  const Value& v = untagged();
  if (v.kind == Kind::kText) return v.bytes;
  return std::nullopt;
}

Value decode(const uint8_t* data, size_t size) {
  Reader reader(data, size);
  Value v = reader.value(0);
  if (!reader.done()) throw std::runtime_error("trailing bytes after CBOR item");
  while (v.kind == Value::Kind::kTag && v.integer == kSelfDescribedTag) {
    Value inner = std::move(v.items[0]);
    v = std::move(inner);
  }
  return v;
}

size_t NumericArray::element_size() const { return type_size(type); }

size_t NumericArray::elements() const { return data.size() / element_size(); }

std::optional<NumericArray> numeric_array(const Value& value) {
  const Value* v = &value;
  std::vector<uint64_t> shape;
  if (v->kind == Value::Kind::kTag && v->integer == kMultiDimTag) {
    const Value& pair = v->items[0];
    if (pair.kind != Value::Kind::kArray || pair.items.size() != 2 ||
        pair.items[0].kind != Value::Kind::kArray)
      return std::nullopt;
    for (const auto& dim : pair.items[0].items) {
      const auto n = dim.as_int();
      if (!n || *n < 0) return std::nullopt;
      shape.push_back(static_cast<uint64_t>(*n));
    }
    v = &pair.items[1];
  }
  std::optional<NumericArray> array;
  if (v->kind == Value::Kind::kTag && v->items[0].kind == Value::Kind::kBytes) {
    array = typed_array(v->integer, v->items[0].bytes);
  } else if (v->kind == Value::Kind::kArray) {
    NumericArray plain;
    plain.data.resize(v->items.size() * sizeof(double));
    for (size_t i = 0; i < v->items.size(); ++i) {
      const auto x = v->items[i].as_double();
      if (!x) return std::nullopt;
      std::memcpy(plain.data.data() + i * sizeof(double), &*x, sizeof(double));
    }
    plain.shape = {v->items.size()};
    array = std::move(plain);
  }
  if (!array) return std::nullopt;
  if (!shape.empty()) {
    uint64_t total = 1;
    for (auto n : shape) total *= n;
    if (total != array->elements()) return std::nullopt;
    array->shape = std::move(shape);
  }
  return array;
}

}  // namespace dectris::cbor
