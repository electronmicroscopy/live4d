#include <memory>
#include <string>
#include <variant>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/python/core/component_util.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constants.h"
#include "decompress_gpu_operator.hpp"
#include "zmq_rx_operator.hpp"

namespace py = pybind11;
using pybind11::literals::operator""_a;

namespace holoscan::ops {

std::shared_ptr<ZmqRxOp> make_zmq_rx_op(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
    const std::string& endpoint = dectris::kDefaultZmqEndpoint,
    int64_t batch_size = static_cast<int64_t>(dectris::kDefaultBatchSize),
    int64_t detector_width = static_cast<int64_t>(dectris::kDefaultDetectorWidth),
    int64_t detector_height = static_cast<int64_t>(dectris::kDefaultDetectorHeight),
    const std::string& dtype = dectris::kDefaultPixelDtype,
    int64_t n_connections = 1,
    int64_t expected_images = 0,
    const std::string& name = "zmq_rx") {
  auto op = std::make_shared<ZmqRxOp>(
      ArgList{Arg{"endpoint", endpoint}, Arg{"batch_size", batch_size},
              Arg{"detector_width", detector_width},
              Arg{"detector_height", detector_height}, Arg{"dtype", dtype},
              Arg{"n_connections", n_connections},
              Arg{"expected_images", expected_images}});
  init_operator_base(op.get(), fragment_or_subgraph, name);
  return op;
}

std::shared_ptr<DecompressGpuOp> make_decompress_gpu_op(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
    int64_t detector_width = static_cast<int64_t>(dectris::kDefaultDetectorWidth),
    int64_t detector_height = static_cast<int64_t>(dectris::kDefaultDetectorHeight),
    const std::string& dtype = dectris::kDefaultPixelDtype,
    const std::string& name = "decompress_gpu") {
  auto op = std::make_shared<DecompressGpuOp>(
      ArgList{Arg{"detector_width", detector_width},
              Arg{"detector_height", detector_height}, Arg{"dtype", dtype}});
  init_operator_base(op.get(), fragment_or_subgraph, name);
  return op;
}

}  // namespace holoscan::ops

PYBIND11_MODULE(dectris_holoscan_ops, m) {
  py::module_::import("holoscan.core");
  m.attr("__version__") = holoscan::ops::dectris::kOperatorsVersion;

  py::class_<holoscan::ops::ZmqRxOp,
             holoscan::Operator,
             std::shared_ptr<holoscan::ops::ZmqRxOp>>(
      m,
      "ZmqRxOp",
      "Receive DECTRIS Stream2 messages over ZMQ and emit pre-extracted compressed batches.")
      .def(py::init(&holoscan::ops::make_zmq_rx_op),
           "fragment"_a,
           "endpoint"_a = holoscan::ops::dectris::kDefaultZmqEndpoint,
           "batch_size"_a = static_cast<int64_t>(holoscan::ops::dectris::kDefaultBatchSize),
           "detector_width"_a = static_cast<int64_t>(holoscan::ops::dectris::kDefaultDetectorWidth),
           "detector_height"_a = static_cast<int64_t>(holoscan::ops::dectris::kDefaultDetectorHeight),
           "dtype"_a = holoscan::ops::dectris::kDefaultPixelDtype,
           "n_connections"_a = static_cast<int64_t>(1),
           "expected_images"_a = static_cast<int64_t>(0),
           "name"_a = "zmq_rx");

  py::class_<holoscan::ops::DecompressGpuOp,
             holoscan::Operator,
             std::shared_ptr<holoscan::ops::DecompressGpuOp>>(
      m,
      "DecompressGpuOp",
      "Decompress BSLZ4 compressed batches on the GPU and emit a TensorMap.")
      .def(py::init(&holoscan::ops::make_decompress_gpu_op),
           "fragment"_a,
           "detector_width"_a = static_cast<int64_t>(holoscan::ops::dectris::kDefaultDetectorWidth),
           "detector_height"_a = static_cast<int64_t>(holoscan::ops::dectris::kDefaultDetectorHeight),
           "dtype"_a = holoscan::ops::dectris::kDefaultPixelDtype,
           "name"_a = "decompress_gpu");
}
