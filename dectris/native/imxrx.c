/* imxrx: see imxrx.h. */
#define _GNU_SOURCE
#include "imxrx.h"

#include <arpa/inet.h>
#include <endian.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

#define DEFAULT_RECV_DEPTH 1024
#define DEFAULT_SEND_DEPTH 64
#define DEFAULT_SIGNAL_EVERY 16
#define MIN_INLINE 64
#define POST_CHUNK 64
#define POLL_CHUNK 32

/* Work request ids: a signaled WRITE carries the number of WRITEs it completes (itself and the
 * unsignaled ones before it), an unsignaled one 0. RECEIVEs have their own CQ. */
#define WRID_RECV 0

struct imx_ctx {
  struct ibv_context *verbs;
  struct ibv_pd *pd;
  imx_ctx_info info;
};

struct imx_mr {
  struct ibv_mr *mr;
};

struct imx_qp {
  imx_ctx *ctx;
  struct ibv_cq *send_cq, *recv_cq;
  struct ibv_qp *qp;
  uint32_t psn;
  uint32_t max_inline;
  int recv_depth, send_depth, signal_every;
  int since_signal; /* WRITEs posted since the last signaled one */
  imx_stats stats;
  char cause[256]; /* first failed completion other than a flush */
  char flush[256]; /* first flush */
  char errors[520];
};

static __thread char g_err[512];

const char *imx_last_error(void) { return g_err; }

/* Set the thread's error and return -err. */
static int fail(int err, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static int fail(int err, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
  if (err && n >= 0 && (size_t)n < sizeof(g_err))
    snprintf(g_err + n, sizeof(g_err) - n, ": %s", strerror(err));
  return -(err ? err : EINVAL);
}

/* ---- Device context ------------------------------------------------------------------------ */

/* The GID of an address: an IPv4 address as an IPv4-mapped IPv6 address, as RoCE v2 has it. */
static int addr_to_gid(const char *addr, uint8_t gid[16]) {
  struct in_addr v4;
  if (inet_pton(AF_INET, addr, &v4) == 1) {
    memset(gid, 0, 16);
    gid[10] = gid[11] = 0xff;
    memcpy(gid + 12, &v4, 4);
    return 0;
  }
  return inet_pton(AF_INET6, addr, gid) == 1 ? 0 : -1;
}

static int is_v4_mapped(const uint8_t gid[16]) {
  static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  return memcmp(gid, prefix, sizeof(prefix)) == 0;
}

/* GID index on `port` of `gid` (want != NULL) or of the first RoCE v2 IPv4 entry, preferring the
 * RoCE v2 entry: v1 and v2 entries of one address carry the same GID. -1 if none. */
static int find_gid(struct ibv_context *verbs, uint8_t port, int table_len, const uint8_t *want) {
  int any = -1;
  for (int i = 0; i < table_len; ++i) {
    struct ibv_gid_entry e;
    if (ibv_query_gid_ex(verbs, port, i, &e, 0) != 0) continue;
    if (want ? memcmp(e.gid.raw, want, 16) != 0 : !is_v4_mapped(e.gid.raw)) continue;
    if (e.gid_type == IBV_GID_TYPE_ROCE_V2) return i;
    if (any < 0 && want) any = i;
  }
  return any;
}

/* Fill ctx->info from `verbs` if the configuration matches one of its ports. */
static int match_device(struct ibv_context *verbs, const imx_ctx_config *cfg, const uint8_t *want,
                        imx_ctx_info *info) {
  struct ibv_device_attr dev;
  if (ibv_query_device(verbs, &dev) != 0) return 0;
  for (uint8_t port = 1; port <= dev.phys_port_cnt; ++port) {
    if (cfg->port && port != cfg->port) continue;
    struct ibv_port_attr pa;
    if (ibv_query_port(verbs, port, &pa) != 0) continue;
    const int ethernet = pa.link_layer == IBV_LINK_LAYER_ETHERNET;
    int index = cfg->gid_index;
    if (index < 0) {
      if (want || ethernet) index = find_gid(verbs, port, pa.gid_tbl_len, want);
      else index = 0;
    }
    if (index < 0) continue;
    union ibv_gid gid;
    if (ibv_query_gid(verbs, port, index, &gid) != 0) continue;
    if (want && memcmp(gid.raw, want, 16) != 0) continue;
    snprintf(info->device, sizeof(info->device), "%s", ibv_get_device_name(verbs->device));
    info->port = port;
    info->gid_index = index;
    memcpy(info->gid, gid.raw, 16);
    info->lid = pa.lid;
    info->active_mtu = (uint8_t)pa.active_mtu;
    info->ethernet = ethernet;
    return 1;
  }
  return 0;
}

int imx_ctx_open(const imx_ctx_config *cfg, imx_ctx **out) {
  *out = NULL;
  if (!cfg->local_addr && !cfg->device) return fail(EINVAL, "need a local address or a device");
  uint8_t want[16];
  if (cfg->local_addr && addr_to_gid(cfg->local_addr, want) != 0)
    return fail(EINVAL, "invalid address %s", cfg->local_addr);

  imx_ctx *ctx = calloc(1, sizeof(*ctx));
  if (!ctx) return fail(ENOMEM, "imx_ctx_open");
  int n = 0;
  struct ibv_device **list = ibv_get_device_list(&n);
  if (!list) {
    free(ctx);
    return fail(errno, "no RDMA devices");
  }
  for (int i = 0; i < n && !ctx->verbs; ++i) {
    if (cfg->device && strcmp(ibv_get_device_name(list[i]), cfg->device) != 0) continue;
    struct ibv_context *verbs = ibv_open_device(list[i]);
    if (!verbs) continue;
    if (match_device(verbs, cfg, cfg->local_addr ? want : NULL, &ctx->info)) ctx->verbs = verbs;
    else ibv_close_device(verbs);
  }
  ibv_free_device_list(list);
  if (!ctx->verbs) {
    free(ctx);
    return fail(ENODEV, "no RDMA port %s%s%s%s", cfg->local_addr ? "with address " : "",
                cfg->local_addr ? cfg->local_addr : "", cfg->device ? " on device " : "",
                cfg->device ? cfg->device : "");
  }
  ctx->pd = ibv_alloc_pd(ctx->verbs);
  if (!ctx->pd) {
    int err = errno;
    ibv_close_device(ctx->verbs);
    free(ctx);
    return fail(err, "ibv_alloc_pd");
  }
  *out = ctx;
  return 0;
}

void imx_ctx_info_get(const imx_ctx *ctx, imx_ctx_info *info) { *info = ctx->info; }

void imx_ctx_close(imx_ctx *ctx) {
  if (!ctx) return;
  ibv_dealloc_pd(ctx->pd);
  ibv_close_device(ctx->verbs);
  free(ctx);
}

/* ---- Memory regions ------------------------------------------------------------------------ */

static int access_flags(int access) {
  int flags = IBV_ACCESS_LOCAL_WRITE;
  if (access & IMX_ACCESS_REMOTE_WRITE) flags |= IBV_ACCESS_REMOTE_WRITE;
  if (access & IMX_ACCESS_REMOTE_READ) flags |= IBV_ACCESS_REMOTE_READ;
  return flags;
}

int imx_mr_register(imx_ctx *ctx, void *addr, size_t length, int access, imx_mr **out) {
  *out = NULL;
  imx_mr *mr = calloc(1, sizeof(*mr));
  if (!mr) return fail(ENOMEM, "imx_mr_register");
  mr->mr = ibv_reg_mr(ctx->pd, addr, length, access_flags(access));
  if (!mr->mr) {
    int err = errno;
    free(mr);
    return fail(err, "ibv_reg_mr(%p, %zu) (GPU memory needs nvidia_peermem)", addr, length);
  }
  *out = mr;
  return 0;
}

int imx_mr_register_dmabuf(imx_ctx *ctx, int fd, uint64_t offset, size_t length, uint64_t iova,
                           int access, imx_mr **out) {
  *out = NULL;
  imx_mr *mr = calloc(1, sizeof(*mr));
  if (!mr) return fail(ENOMEM, "imx_mr_register_dmabuf");
  mr->mr = ibv_reg_dmabuf_mr(ctx->pd, offset, length, iova, fd, access_flags(access));
  if (!mr->mr) {
    int err = errno;
    free(mr);
    return fail(err, "ibv_reg_dmabuf_mr(fd %d, offset %llu, %zu)", fd,
                (unsigned long long)offset, length);
  }
  *out = mr;
  return 0;
}

void imx_mr_info_get(const imx_mr *mr, imx_mr_info *info) {
  info->addr = (uint64_t)(uintptr_t)mr->mr->addr;
  info->length = mr->mr->length;
  info->lkey = mr->mr->lkey;
  info->rkey = mr->mr->rkey;
}

void imx_mr_deregister(imx_mr *mr) {
  if (!mr) return;
  ibv_dereg_mr(mr->mr);
  free(mr);
}

/* ---- Queue pair ---------------------------------------------------------------------------- */

int imx_qp_create(imx_ctx *ctx, const imx_qp_config *cfg, imx_qp **out) {
  *out = NULL;
  imx_qp *qp = calloc(1, sizeof(*qp));
  if (!qp) return fail(ENOMEM, "imx_qp_create");
  qp->ctx = ctx;
  qp->recv_depth = cfg && cfg->recv_depth > 0 ? cfg->recv_depth : DEFAULT_RECV_DEPTH;
  qp->send_depth = cfg && cfg->send_depth > 0 ? cfg->send_depth : DEFAULT_SEND_DEPTH;
  qp->signal_every = cfg && cfg->signal_every > 0 ? cfg->signal_every : DEFAULT_SIGNAL_EVERY;
  if (qp->signal_every > qp->send_depth) qp->signal_every = qp->send_depth;
  int err;

  /* Separate CQs: send completions can always be drained, RECEIVEs only as far as the caller has
   * room for their immediates. */
  qp->send_cq = ibv_create_cq(ctx->verbs, qp->send_depth, NULL, NULL, 0);
  qp->recv_cq = ibv_create_cq(ctx->verbs, qp->recv_depth, NULL, NULL, 0);
  if (!qp->send_cq || !qp->recv_cq) {
    err = fail(errno, "ibv_create_cq(%d, %d)", qp->send_depth, qp->recv_depth);
    goto out;
  }
  struct ibv_qp_init_attr init = {
      .send_cq = qp->send_cq,
      .recv_cq = qp->recv_cq,
      .cap = {.max_send_wr = (uint32_t)qp->send_depth,
              .max_recv_wr = (uint32_t)qp->recv_depth,
              .max_send_sge = 1,
              .max_recv_sge = 1,
              .max_inline_data = MIN_INLINE},
      .qp_type = IBV_QPT_RC,
      .sq_sig_all = 0,
  };
  qp->qp = ibv_create_qp(ctx->pd, &init);
  if (!qp->qp) {
    err = fail(errno, "ibv_create_qp(send %d, recv %d)", qp->send_depth, qp->recv_depth);
    goto out;
  }
  qp->max_inline = init.cap.max_inline_data;

  /* Remote access must be enabled here: without it the peer's RDMA writes fail with a remote
   * access error, which on our side shows only as flushed RECEIVEs. */
  struct ibv_qp_attr attr = {
      .qp_state = IBV_QPS_INIT,
      .pkey_index = 0,
      .port_num = ctx->info.port,
      .qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
  };
  if (ibv_modify_qp(qp->qp, &attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    err = fail(errno, "QP %u RESET -> INIT", qp->qp->qp_num);
    goto out;
  }
  if (getrandom(&qp->psn, sizeof(qp->psn), 0) != sizeof(qp->psn)) qp->psn = (uint32_t)rand();
  qp->psn &= 0xFFFFFF;
  *out = qp;
  return 0;
out:
  imx_qp_destroy(qp);
  return err;
}

void imx_qp_local_endpoint(const imx_qp *qp, imx_endpoint *ep) {
  const imx_ctx_info *info = &qp->ctx->info;
  ep->qp_num = qp->qp->qp_num;
  ep->psn = qp->psn;
  ep->lid = info->lid;
  ep->mtu = info->active_mtu;
  memcpy(ep->gid, info->gid, 16);
}

int imx_qp_connect(imx_qp *qp, const imx_endpoint *remote) {
  const imx_ctx_info *info = &qp->ctx->info;
  const uint8_t mtu = remote->mtu < info->active_mtu ? remote->mtu : info->active_mtu;
  if (mtu < IBV_MTU_256 || mtu > IBV_MTU_4096)
    return fail(EINVAL, "unsupported path MTU %u (remote %u, local %u)", mtu, remote->mtu,
                info->active_mtu);

  struct ibv_qp_attr attr = {
      .qp_state = IBV_QPS_RTR,
      .path_mtu = (enum ibv_mtu)mtu,
      .dest_qp_num = remote->qp_num,
      .rq_psn = remote->psn,
      .max_dest_rd_atomic = 1,
      .min_rnr_timer = 12,
      .ah_attr = {.port_num = info->port, .sl = 0},
  };
  if (info->ethernet) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.sgid_index = (uint8_t)info->gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    memcpy(attr.ah_attr.grh.dgid.raw, remote->gid, 16);
  } else {
    attr.ah_attr.dlid = remote->lid;
  }
  if (ibv_modify_qp(qp->qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    return fail(errno, "QP %u INIT -> RTR (remote QP %u, path MTU %u)", qp->qp->qp_num,
                remote->qp_num, mtu);

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7; /* infinite: the peer waits for RECEIVEs instead of failing */
  attr.sq_psn = qp->psn;
  attr.max_rd_atomic = 1;
  if (ibv_modify_qp(qp->qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
    return fail(errno, "QP %u RTR -> RTS", qp->qp->qp_num);
  return 0;
}

int imx_qp_state(imx_qp *qp) {
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init;
  if (ibv_query_qp(qp->qp, &attr, IBV_QP_STATE, &init) != 0)
    return fail(errno, "ibv_query_qp(%u)", qp->qp->qp_num);
  return attr.qp_state;
}

int imx_qp_post_recvs(imx_qp *qp, int n) {
  const int room = qp->recv_depth - qp->stats.recvs_outstanding;
  if (n > room) n = room;
  struct ibv_recv_wr wrs[POST_CHUNK];
  int posted = 0;
  while (posted < n) {
    const int k = n - posted < POST_CHUNK ? n - posted : POST_CHUNK;
    for (int i = 0; i < k; ++i) {
      wrs[i] = (struct ibv_recv_wr){.wr_id = WRID_RECV, .sg_list = NULL, .num_sge = 0,
                                    .next = i + 1 < k ? &wrs[i + 1] : NULL};
    }
    struct ibv_recv_wr *bad = NULL;
    int rc = ibv_post_recv(qp->qp, wrs, &bad);
    const int done = rc == 0 ? k : (int)(bad - wrs);
    posted += done;
    qp->stats.recvs_outstanding += done;
    qp->stats.recvs_posted += done;
    if (rc != 0) {
      fail(rc, "ibv_post_recv: posted %d of %d", posted, n);
      break;
    }
  }
  return posted;
}

static void record_error(imx_qp *qp, const struct ibv_wc *wc, const char *what) {
  const int flush = wc->status == IBV_WC_WR_FLUSH_ERR;
  char *slot = flush ? qp->flush : qp->cause;
  ++qp->stats.failed;
  if (flush) ++qp->stats.flushed;
  if (slot[0]) return;
  snprintf(slot, sizeof(qp->cause), "%s on QP %u: %s %s (%d), vendor_err %#x, opcode %d",
           flush ? "first flushed work request (the QP is in the error state)"
                 : "first failed completion",
           qp->qp->qp_num, what, ibv_wc_status_str(wc->status), (int)wc->status, wc->vendor_err,
           (int)wc->opcode);
}

int imx_qp_poll(imx_qp *qp, uint32_t *imms, int max_imm, imx_poll_result *res) {
  memset(res, 0, sizeof(*res));
  struct ibv_wc wc[POLL_CHUNK];
  int got;
  do {
    got = ibv_poll_cq(qp->send_cq, POLL_CHUNK, wc);
    if (got < 0) return fail(EIO, "ibv_poll_cq (send) on QP %u", qp->qp->qp_num);
    for (int i = 0; i < got; ++i) {
      const int covered = (int)wc[i].wr_id;
      qp->stats.writes_outstanding -= covered;
      qp->stats.writes_completed += covered;
      res->n_tx_done += covered;
      if (wc[i].status != IBV_WC_SUCCESS) {
        record_error(qp, &wc[i], "WRITE");
        ++res->n_failed;
      }
    }
  } while (got == POLL_CHUNK);

  while (res->n_imm < max_imm) {
    /* Never take more completions than immediates fit. */
    const int want = max_imm - res->n_imm < POLL_CHUNK ? max_imm - res->n_imm : POLL_CHUNK;
    got = ibv_poll_cq(qp->recv_cq, want, wc);
    if (got < 0) return fail(EIO, "ibv_poll_cq (recv) on QP %u", qp->qp->qp_num);
    for (int i = 0; i < got; ++i) {
      const struct ibv_wc *w = &wc[i];
      --qp->stats.recvs_outstanding;
      if (w->status == IBV_WC_SUCCESS && w->opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
          (w->wc_flags & IBV_WC_WITH_IMM)) {
        imms[res->n_imm++] = be32toh(w->imm_data);
        ++qp->stats.imm_completions;
      } else {
        /* A SEND instead of a WRITE_WITH_IMM is as fatal as a failure: nothing expects it. */
        record_error(qp, w, w->status == IBV_WC_SUCCESS ? "unexpected RECEIVE" : "RECEIVE");
        ++res->n_failed;
      }
    }
    if (got < want) break;
  }
  return 0;
}

static int post_write(imx_qp *qp, struct ibv_sge *sge, uint64_t remote_addr, uint32_t rkey,
                      int with_imm, uint32_t imm, unsigned flags) {
  if (qp->stats.writes_outstanding >= qp->send_depth) return 0;
  /* Unsignaled WRITEs keep their send queue slot until a later signaled one completes, so
   * signal at least every signal_every and always before the queue would fill up. */
  const int count = qp->since_signal + 1;
  const int signal =
      count >= qp->signal_every || qp->stats.writes_outstanding + 1 >= qp->send_depth;
  struct ibv_send_wr wr = {
      .wr_id = signal ? (uint64_t)count : 0,
      .sg_list = sge,
      .num_sge = 1,
      .opcode = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE,
      .send_flags = flags | (signal ? IBV_SEND_SIGNALED : 0),
      .wr.rdma = {.remote_addr = remote_addr, .rkey = rkey},
  };
  if (with_imm) wr.imm_data = htobe32(imm);
  struct ibv_send_wr *bad = NULL;
  int rc = ibv_post_send(qp->qp, &wr, &bad);
  if (rc != 0) return fail(rc, "ibv_post_send on QP %u", qp->qp->qp_num);
  qp->since_signal = signal ? 0 : count;
  ++qp->stats.writes_outstanding;
  ++qp->stats.writes_posted;
  return 1;
}

int imx_qp_write(imx_qp *qp, const imx_mr *mr, uint64_t offset, uint32_t length,
                 uint64_t remote_addr, uint32_t rkey, int with_imm, uint32_t imm) {
  if (offset + length > mr->mr->length)
    return fail(EINVAL, "write of %u bytes at offset %llu exceeds the region (%zu bytes)", length,
                (unsigned long long)offset, mr->mr->length);
  struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)mr->mr->addr + offset,
                        .length = length,
                        .lkey = mr->mr->lkey};
  return post_write(qp, &sge, remote_addr, rkey, with_imm, imm, 0);
}

int imx_qp_write_inline(imx_qp *qp, const void *data, uint32_t length, uint64_t remote_addr,
                        uint32_t rkey) {
  if (length > qp->max_inline)
    return fail(EINVAL, "inline write of %u bytes exceeds the limit of %u", length,
                qp->max_inline);
  struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)data, .length = length, .lkey = 0};
  return post_write(qp, &sge, remote_addr, rkey, 0, 0, IBV_SEND_INLINE);
}

void imx_qp_stats_get(const imx_qp *qp, imx_stats *stats) { *stats = qp->stats; }

const char *imx_qp_errors(const imx_qp *qp) {
  imx_qp *q = (imx_qp *)qp;
  snprintf(q->errors, sizeof(q->errors), "%s%s%s", q->cause, q->cause[0] && q->flush[0] ? "; " : "",
           q->flush);
  return q->errors;
}

void imx_qp_destroy(imx_qp *qp) {
  if (!qp) return;
  if (qp->qp) ibv_destroy_qp(qp->qp);
  if (qp->send_cq) ibv_destroy_cq(qp->send_cq);
  if (qp->recv_cq) ibv_destroy_cq(qp->recv_cq);
  free(qp);
}
