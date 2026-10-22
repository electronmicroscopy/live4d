// Python binding of imxrx (module imxrdma). Objects keep what they depend on alive: a memory
// region or queue pair holds its context, so the protection domain outlives them.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "imxrx.h"

namespace py = pybind11;
using namespace py::literals;

namespace {

[[noreturn]] void raise(const char *what) {
  throw std::runtime_error(std::string(what) + ": " + imx_last_error());
}

py::bytes gid_bytes(const uint8_t gid[16]) {
  return py::bytes(reinterpret_cast<const char *>(gid), 16);
}

struct Context {
  imx_ctx *ctx = nullptr;
  explicit Context(imx_ctx *c) : ctx(c) {}
  ~Context() { imx_ctx_close(ctx); }
  imx_ctx_info info() const {
    imx_ctx_info i;
    imx_ctx_info_get(ctx, &i);
    return i;
  }
};

struct MemoryRegion {
  std::shared_ptr<Context> ctx;
  imx_mr *mr = nullptr;
  imx_mr_info info{};
  MemoryRegion(std::shared_ptr<Context> c, imx_mr *m) : ctx(std::move(c)), mr(m) {
    imx_mr_info_get(mr, &info);
  }
  ~MemoryRegion() { close(); }
  void close() {
    imx_mr_deregister(mr);
    mr = nullptr;
  }
  imx_mr *get() const {
    if (!mr) throw std::runtime_error("memory region is closed");
    return mr;
  }
};

struct QueuePair {
  std::shared_ptr<Context> ctx;
  imx_qp *qp = nullptr;
  QueuePair(std::shared_ptr<Context> c, imx_qp *q) : ctx(std::move(c)), qp(q) {}
  ~QueuePair() { close(); }
  void close() {
    imx_qp_destroy(qp);
    qp = nullptr;
  }
  imx_qp *get() const {
    if (!qp) throw std::runtime_error("queue pair is closed");
    return qp;
  }
};

int access_of(bool remote_write, bool remote_read) {
  return (remote_write ? IMX_ACCESS_REMOTE_WRITE : 0) | (remote_read ? IMX_ACCESS_REMOTE_READ : 0);
}

}  // namespace

PYBIND11_MODULE(imxrdma, m) {
  m.doc() =
      "RDMA verbs for the imagemux receiver on plain libibverbs: one RC queue pair connected out "
      "of band, memory regions, zero-length RECEIVEs and RDMA WRITEs.";

  py::class_<imx_endpoint>(m, "Endpoint",
                           "What a peer needs to connect its RC queue pair to ours.")
      .def(py::init([](uint32_t qp_num, uint32_t psn, uint16_t lid, uint8_t mtu,
                       const py::bytes &gid) {
             const std::string g = gid;
             if (g.size() != 16) throw std::invalid_argument("gid must be 16 bytes");
             imx_endpoint ep{qp_num, psn, lid, mtu, {}};
             std::memcpy(ep.gid, g.data(), 16);
             return ep;
           }),
           "qp_num"_a, "psn"_a, "lid"_a, "mtu"_a, "gid"_a)
      .def_readonly("qp_num", &imx_endpoint::qp_num)
      .def_readonly("psn", &imx_endpoint::psn)
      .def_readonly("lid", &imx_endpoint::lid)
      .def_readonly("mtu", &imx_endpoint::mtu)
      .def_property_readonly("gid", [](const imx_endpoint &e) { return gid_bytes(e.gid); })
      .def("__repr__", [](const imx_endpoint &e) {
        return "Endpoint(qp_num=" + std::to_string(e.qp_num) + ", psn=" + std::to_string(e.psn) +
               ", lid=" + std::to_string(e.lid) + ", mtu=" + std::to_string(e.mtu) + ")";
      });

  py::class_<MemoryRegion, std::shared_ptr<MemoryRegion>>(m, "MemoryRegion")
      .def_property_readonly("address", [](const MemoryRegion &r) { return r.info.addr; })
      .def_property_readonly("length", [](const MemoryRegion &r) { return r.info.length; })
      .def_property_readonly("lkey", [](const MemoryRegion &r) { return r.info.lkey; })
      .def_property_readonly("rkey", [](const MemoryRegion &r) { return r.info.rkey; })
      .def("close", &MemoryRegion::close, "Deregister now instead of on garbage collection.");

  py::class_<QueuePair, std::shared_ptr<QueuePair>>(m, "QueuePair")
      .def_property_readonly("endpoint",
                             [](const QueuePair &q) {
                               imx_endpoint ep;
                               imx_qp_local_endpoint(q.get(), &ep);
                               return ep;
                             })
      .def(
          "connect",
          [](QueuePair &q, const imx_endpoint &remote) {
            if (imx_qp_connect(q.get(), &remote) != 0) raise("connect");
          },
          "remote"_a, "INIT -> RTR -> RTS towards the peer's endpoint.")
      .def_property_readonly("state",
                             [](QueuePair &q) {
                               int s = imx_qp_state(q.get());
                               if (s < 0) raise("state");
                               return s;
                             },
                             "enum ibv_qp_state (3 = RTS, 6 = ERR).")
      .def(
          "post_recvs",
          [](QueuePair &q, int n) {
            imx_qp *qp = q.get();
            py::gil_scoped_release nogil;
            return imx_qp_post_recvs(qp, n);
          },
          "n"_a,
          "Post up to n zero-length RECEIVEs, as many as fit in recv_depth. Returns the number "
          "posted.")
      .def(
          "poll",
          [](QueuePair &q, py::array_t<uint32_t, py::array::c_style> imms) {
            if (imms.ndim() != 1) throw std::invalid_argument("imms must be 1-D uint32");
            imx_qp *qp = q.get();
            uint32_t *out = imms.mutable_data();
            const int max = static_cast<int>(imms.shape(0));
            imx_poll_result res;
            int rc;
            {
              py::gil_scoped_release nogil;
              rc = imx_qp_poll(qp, out, max, &res);
            }
            if (rc != 0) raise("poll");
            return py::make_tuple(res.n_imm, res.n_tx_done, res.n_failed);
          },
          "imms"_a,
          "Drain completions into imms (uint32, its length is the maximum) without blocking. "
          "Returns (n_imm, n_tx_done, n_failed): imms[:n_imm] are the WRITE_WITH_IMM immediates in "
          "arrival order; n_failed > 0 means the connection is unusable (see errors).")
      .def(
          "write",
          [](QueuePair &q, const MemoryRegion &mr, uint64_t offset, uint32_t length,
             uint64_t remote_addr, uint32_t rkey, py::object imm) {
            imx_qp *qp = q.get();
            imx_mr *r = mr.get();
            const bool with_imm = !imm.is_none();
            const uint32_t value = with_imm ? imm.cast<uint32_t>() : 0;
            int rc;
            {
              py::gil_scoped_release nogil;
              rc = imx_qp_write(qp, r, offset, length, remote_addr, rkey, with_imm, value);
            }
            if (rc < 0) raise("write");
            return rc == 1;
          },
          "mr"_a, "offset"_a, "length"_a, "remote_addr"_a, "rkey"_a, "imm"_a = py::none(),
          "RDMA WRITE (WRITE_WITH_IMM if imm is given) from mr to the peer. False if the send "
          "queue is full: poll, then retry.")
      .def(
          "write_inline",
          [](QueuePair &q, const py::bytes &data, uint64_t remote_addr, uint32_t rkey) {
            const std::string d = data;
            int rc = imx_qp_write_inline(q.get(), d.data(), static_cast<uint32_t>(d.size()),
                                         remote_addr, rkey);
            if (rc < 0) raise("write_inline");
            return rc == 1;
          },
          "data"_a, "remote_addr"_a, "rkey"_a,
          "RDMA WRITE of a few bytes carried in the work request (no memory region). False if "
          "the send queue is full.")
      .def(
          "write_u64",
          [](QueuePair &q, uint64_t value, uint64_t remote_addr, uint32_t rkey) {
            int rc = imx_qp_write_inline(q.get(), &value, sizeof(value), remote_addr, rkey);
            if (rc < 0) raise("write_u64");
            return rc == 1;
          },
          "value"_a, "remote_addr"_a, "rkey"_a,
          "RDMA WRITE of a little-endian u64 (the credit word). False if the send queue is full.")
      .def_property_readonly("stats",
                             [](const QueuePair &q) {
                               imx_stats s;
                               imx_qp_stats_get(q.get(), &s);
                               return py::dict("recvs_posted"_a = s.recvs_posted,
                                               "imm_completions"_a = s.imm_completions,
                                               "writes_posted"_a = s.writes_posted,
                                               "writes_completed"_a = s.writes_completed,
                                               "failed"_a = s.failed, "flushed"_a = s.flushed,
                                               "recvs_outstanding"_a = s.recvs_outstanding,
                                               "writes_outstanding"_a = s.writes_outstanding);
                             })
      .def_property_readonly(
          "errors", [](const QueuePair &q) { return std::string(imx_qp_errors(q.get())); },
          "The first failed completion and the first flush, or '' if there was none.")
      .def("close", &QueuePair::close, "Destroy now instead of on garbage collection.");

  py::class_<Context, std::shared_ptr<Context>>(m, "Context",
                                                "An RDMA device port and protection domain.")
      .def(py::init([](py::object local_addr, py::object device, uint8_t port, int gid_index) {
             std::string addr = local_addr.is_none() ? "" : local_addr.cast<std::string>();
             std::string dev = device.is_none() ? "" : device.cast<std::string>();
             imx_ctx_config cfg{local_addr.is_none() ? nullptr : addr.c_str(),
                                device.is_none() ? nullptr : dev.c_str(), port, gid_index};
             imx_ctx *ctx = nullptr;
             if (imx_ctx_open(&cfg, &ctx) != 0) raise("Context");
             return std::make_shared<Context>(ctx);
           }),
           "local_addr"_a = py::none(), "device"_a = py::none(), "port"_a = 0,
           "gid_index"_a = -1,
           "Open the RDMA port of the interface with address local_addr (RoCE v2 GID), or of "
           "device/port.")
      .def_property_readonly("device", [](const Context &c) { return std::string(c.info().device); })
      .def_property_readonly("port", [](const Context &c) { return c.info().port; })
      .def_property_readonly("gid_index", [](const Context &c) { return c.info().gid_index; })
      .def_property_readonly("gid", [](const Context &c) { return gid_bytes(c.info().gid); })
      .def_property_readonly("active_mtu", [](const Context &c) { return c.info().active_mtu; })
      .def(
          "register",
          [](std::shared_ptr<Context> c, uint64_t addr, size_t length, bool remote_write,
             bool remote_read) {
            imx_mr *mr = nullptr;
            if (imx_mr_register(c->ctx, reinterpret_cast<void *>(addr), length,
                                access_of(remote_write, remote_read), &mr) != 0)
              raise("register");
            return std::make_shared<MemoryRegion>(std::move(c), mr);
          },
          "addr"_a, "length"_a, "remote_write"_a = false, "remote_read"_a = false,
          "Register host memory or GPU memory (nvidia_peermem) at addr.")
      .def(
          "register_dmabuf",
          [](std::shared_ptr<Context> c, int fd, uint64_t offset, size_t length, uint64_t iova,
             bool remote_write, bool remote_read) {
            imx_mr *mr = nullptr;
            if (imx_mr_register_dmabuf(c->ctx, fd, offset, length, iova,
                                       access_of(remote_write, remote_read), &mr) != 0)
              raise("register_dmabuf");
            return std::make_shared<MemoryRegion>(std::move(c), mr);
          },
          "fd"_a, "offset"_a, "length"_a, "iova"_a, "remote_write"_a = false,
          "remote_read"_a = false, "Register a dma-buf, addressed by the peer as iova.")
      .def(
          "create_qp",
          [](std::shared_ptr<Context> c, int recv_depth, int send_depth, int signal_every) {
            imx_qp_config cfg{recv_depth, send_depth, signal_every};
            imx_qp *qp = nullptr;
            if (imx_qp_create(c->ctx, &cfg, &qp) != 0) raise("create_qp");
            return std::make_shared<QueuePair>(std::move(c), qp);
          },
          "recv_depth"_a = 1024, "send_depth"_a = 64, "signal_every"_a = 16,
          "Create an RC queue pair in INIT with remote write and read access.");
}
