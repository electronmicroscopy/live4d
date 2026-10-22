#include <holoscan/authoring/register_operator.hpp>
#include "decompress_gpu_operator.hpp"
#include "zmq_rx_operator.hpp"

HOLOSCAN_REGISTER_OPERATOR(holoscan::ops::ZmqRxOp, "holoscan.dectris.ZmqRxOp")
HOLOSCAN_REGISTER_OPERATOR(holoscan::ops::DecompressGpuOp, "holoscan.dectris.DecompressGpuOp")
