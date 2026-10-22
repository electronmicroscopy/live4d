/* imxrx: the RDMA verbs the imagemux receiver needs, on plain libibverbs (rdma-core).
 *
 * One reliable-connected queue pair whose parameters are exchanged out of band (imagemux's TCP
 * handshake, no rdma_cm), memory regions the peer RDMA-writes into, zero-length RECEIVEs consumed
 * by the peer's WRITE_WITH_IMM, and small RDMA WRITEs back to the peer (the credit word).
 *
 * The library never calls CUDA: GPU memory is allocated by the caller and registered like host
 * memory (nvidia_peermem), or through a dma-buf file descriptor.
 *
 * Errors: functions return 0 (or a count) on success and a negative errno on failure, with a
 * description in imx_last_error() (per thread). A queue pair is not thread safe: use it from one
 * thread at a time.
 */
#ifndef IMXRX_H
#define IMXRX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct imx_ctx imx_ctx;
typedef struct imx_mr imx_mr;
typedef struct imx_qp imx_qp;

/* Description of the last failure on this thread. */
const char *imx_last_error(void);

/* ---- Device context ------------------------------------------------------------------------ */

typedef struct imx_ctx_config {
  /* IPv4 address of the RoCE interface: selects the device, port and RoCE v2 GID. May be NULL
   * when `device` is given. */
  const char *local_addr;
  /* RDMA device name (e.g. "roceP5p3s0f1"); NULL = any device holding `local_addr`. */
  const char *device;
  /* Port number (from 1); 0 = from `local_addr`, or 1. */
  uint8_t port;
  /* GID table index; -1 = from `local_addr`, or the first RoCE v2 IPv4 entry of the port. */
  int gid_index;
} imx_ctx_config;

typedef struct imx_ctx_info {
  char device[64];
  uint8_t port;
  int gid_index;
  uint8_t gid[16];
  uint16_t lid;
  uint8_t active_mtu; /* enum ibv_mtu: 1 = 256 B ... 5 = 4096 B */
  int ethernet;       /* RoCE: address the peer by GID; InfiniBand: by LID */
} imx_ctx_info;

int imx_ctx_open(const imx_ctx_config *cfg, imx_ctx **out);
void imx_ctx_info_get(const imx_ctx *ctx, imx_ctx_info *info);
/* Memory regions and queue pairs must be released first. */
void imx_ctx_close(imx_ctx *ctx);

/* ---- Memory regions ------------------------------------------------------------------------ */

enum {
  IMX_ACCESS_REMOTE_WRITE = 1 << 0,
  IMX_ACCESS_REMOTE_READ = 1 << 1,
};

typedef struct imx_mr_info {
  uint64_t addr;
  uint64_t length;
  uint32_t lkey;
  uint32_t rkey;
} imx_mr_info;

/* Register [addr, addr + length): host memory, or GPU memory with nvidia_peermem loaded. Always
 * local-writable. Relaxed ordering is never enabled: the peer's last write (WRITE_WITH_IMM) must
 * not overtake the frame data before it. */
int imx_mr_register(imx_ctx *ctx, void *addr, size_t length, int access, imx_mr **out);
/* Register `length` bytes at `offset` of a dma-buf (e.g. from cuMemGetHandleForAddressRange),
 * addressed by the peer as `iova`. */
int imx_mr_register_dmabuf(imx_ctx *ctx, int fd, uint64_t offset, size_t length, uint64_t iova,
                           int access, imx_mr **out);
void imx_mr_info_get(const imx_mr *mr, imx_mr_info *info);
void imx_mr_deregister(imx_mr *mr);

/* ---- Queue pair ---------------------------------------------------------------------------- */

typedef struct imx_endpoint {
  uint32_t qp_num;
  uint32_t psn;
  uint16_t lid;
  uint8_t mtu; /* enum ibv_mtu */
  uint8_t gid[16];
} imx_endpoint;

typedef struct imx_qp_config {
  int recv_depth;   /* RECEIVEs that may be posted at once (default 1024) */
  int send_depth;   /* RDMA WRITEs in flight (default 64) */
  int signal_every; /* request a send completion every n WRITEs (default 16, <= send_depth) */
} imx_qp_config;

typedef struct imx_stats {
  uint64_t recvs_posted;
  uint64_t imm_completions;  /* WRITE_WITH_IMM received */
  uint64_t writes_posted;
  uint64_t writes_completed; /* WRITEs whose slot in the send queue is free again */
  uint64_t failed;           /* failed or unexpected completions, flushes included */
  uint64_t flushed;          /* of which flushes (the QP is in the error state) */
  int recvs_outstanding;
  int writes_outstanding;
} imx_stats;

typedef struct imx_poll_result {
  int n_imm;      /* immediates stored */
  int n_tx_done;  /* WRITEs completed */
  int n_failed;   /* failed or unexpected completions: the connection is unusable */
} imx_poll_result;

/* Create an RC queue pair in INIT, with remote write and read access. */
int imx_qp_create(imx_ctx *ctx, const imx_qp_config *cfg, imx_qp **out);
/* What the peer needs to connect its queue pair to ours. */
void imx_qp_local_endpoint(const imx_qp *qp, imx_endpoint *ep);
/* INIT -> RTR -> RTS towards the peer's endpoint. The path MTU is the smaller of both. */
int imx_qp_connect(imx_qp *qp, const imx_endpoint *remote);
/* enum ibv_qp_state, or a negative errno. */
int imx_qp_state(imx_qp *qp);
/* Post up to `n` zero-length RECEIVEs, as many as fit in recv_depth. Returns the number posted. */
int imx_qp_post_recvs(imx_qp *qp, int n);
/* Drain completions without blocking, until `max_imm` immediates are stored in `imms` (host
 * order) or none are left. */
int imx_qp_poll(imx_qp *qp, uint32_t *imms, int max_imm, imx_poll_result *res);
/* RDMA WRITE `length` bytes at `offset` of `mr` to the peer's `remote_addr`; a WRITE_WITH_IMM
 * carrying `imm` if `with_imm`. Returns 1 if posted, 0 if the send queue is full (retry after a
 * poll), or a negative errno. */
int imx_qp_write(imx_qp *qp, const imx_mr *mr, uint64_t offset, uint32_t length,
                 uint64_t remote_addr, uint32_t rkey, int with_imm, uint32_t imm);
/* As imx_qp_write, the data copied into the work request (no memory region needed; `length` up
 * to the QP's inline limit, at least 64). */
int imx_qp_write_inline(imx_qp *qp, const void *data, uint32_t length, uint64_t remote_addr,
                        uint32_t rkey);
void imx_qp_stats_get(const imx_qp *qp, imx_stats *stats);
/* The first failed completion and the first flush, or "" if there was none. */
const char *imx_qp_errors(const imx_qp *qp);
void imx_qp_destroy(imx_qp *qp);

#ifdef __cplusplus
}
#endif

#endif
