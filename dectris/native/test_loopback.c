/* Two queue pairs on one RDMA port, connected to each other: a sender that writes frames and
 * announces them by WRITE_WITH_IMM, and a receiver that returns a credit word, as imagemux and
 * the pipeline do.
 *
 *   imxrx_loopback LOCAL_ADDR [FRAMES]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "imxrx.h"

#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                  \
      fprintf(stderr, __VA_ARGS__);                                         \
      fprintf(stderr, " (%s)\n", imx_last_error());                         \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

enum { RING = 1 << 20, FRAME = 4096, SLOTS = RING / FRAME, DEPTH = 256 };

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Poll until `want` immediates arrived on `qp` (or a send completed if want == 0). */
static int poll_for(imx_qp *qp, uint32_t *imms, int want, imx_poll_result *total) {
  memset(total, 0, sizeof(*total));
  double deadline = now() + 5;
  while (now() < deadline) {
    imx_poll_result r;
    CHECK(imx_qp_poll(qp, imms + total->n_imm, want - total->n_imm, &r) == 0, "poll");
    total->n_imm += r.n_imm;
    total->n_tx_done += r.n_tx_done;
    total->n_failed += r.n_failed;
    if (total->n_failed) return 0;
    if (want ? total->n_imm == want : total->n_tx_done > 0) return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s LOCAL_ADDR [FRAMES]\n", argv[0]);
    return 2;
  }
  const int frames = argc > 2 ? atoi(argv[2]) : 100000;

  imx_ctx_config cfg = {.local_addr = argv[1], .gid_index = -1};
  imx_ctx *ctx;
  CHECK(imx_ctx_open(&cfg, &ctx) == 0, "open %s", argv[1]);
  imx_ctx_info info;
  imx_ctx_info_get(ctx, &info);
  printf("device %s port %u gid index %d mtu %u\n", info.device, info.port, info.gid_index,
         info.active_mtu);

  uint8_t *ring = aligned_alloc(4096, RING), *src = aligned_alloc(4096, RING);
  uint64_t *credit_word = aligned_alloc(64, 64);
  *credit_word = 0;
  imx_mr *ring_mr, *src_mr, *credit_mr;
  CHECK(imx_mr_register(ctx, ring, RING, IMX_ACCESS_REMOTE_WRITE, &ring_mr) == 0, "ring");
  CHECK(imx_mr_register(ctx, src, RING, 0, &src_mr) == 0, "src");
  CHECK(imx_mr_register(ctx, credit_word, 64, IMX_ACCESS_REMOTE_WRITE, &credit_mr) == 0, "cred");
  imx_mr_info ring_info, credit_info;
  imx_mr_info_get(ring_mr, &ring_info);
  imx_mr_info_get(credit_mr, &credit_info);

  imx_qp_config qcfg = {.recv_depth = DEPTH, .send_depth = 64, .signal_every = 16};
  imx_qp *rx, *tx;
  CHECK(imx_qp_create(ctx, &qcfg, &rx) == 0, "rx qp");
  CHECK(imx_qp_create(ctx, &qcfg, &tx) == 0, "tx qp");
  imx_endpoint rx_ep, tx_ep;
  imx_qp_local_endpoint(rx, &rx_ep);
  imx_qp_local_endpoint(tx, &tx_ep);
  CHECK(imx_qp_connect(rx, &tx_ep) == 0, "connect rx");
  CHECK(imx_qp_connect(tx, &rx_ep) == 0, "connect tx");
  CHECK(imx_qp_state(rx) == 3 && imx_qp_state(tx) == 3, "both QPs in RTS");
  CHECK(imx_qp_post_recvs(rx, DEPTH + 10) == DEPTH, "post recvs capped at the depth");

  /* 1. One frame: data lands before its immediate, the immediate is in host order. */
  for (int i = 0; i < FRAME; ++i) src[i] = (uint8_t)(i * 7 + 1);
  CHECK(imx_qp_write(tx, src_mr, 0, FRAME, ring_info.addr, ring_info.rkey, 1, 0xA1B2C3D4u) == 1,
        "write imm");
  uint32_t imms[DEPTH];
  imx_poll_result r;
  CHECK(poll_for(rx, imms, 1, &r) && imms[0] == 0xA1B2C3D4u, "immediate %#x", imms[0]);
  CHECK(memcmp(ring, src, FRAME) == 0, "frame data");

  /* 2. The credit word, written inline. */
  CHECK(imx_qp_write_inline(rx, &(uint64_t){0x1122334455667788ull}, 8, credit_info.addr,
                            credit_info.rkey) == 1, "credit");
  double deadline = now() + 5;
  while (*(volatile uint64_t *)credit_word != 0x1122334455667788ull && now() < deadline) {
    imx_qp_poll(rx, imms, 0, &r);
  }
  CHECK(*credit_word == 0x1122334455667788ull, "credit word %#llx",
        (unsigned long long)*credit_word);

  /* 3. Throughput: the sender keeps at most DEPTH/2 frames in flight, the receiver re-posts a
   * RECEIVE per completion and returns credit, sequence numbers in order. */
  imx_qp_post_recvs(rx, 1);
  uint32_t sent = 0, received = 0, acked = 0;
  double t0 = now();
  while (received < (uint32_t)frames) {
    while (sent < (uint32_t)frames && sent - received < DEPTH / 2) {
      int rc = imx_qp_write(tx, src_mr, (uint64_t)(sent % SLOTS) * FRAME, 128,
                            ring_info.addr + (uint64_t)(sent % SLOTS) * FRAME, ring_info.rkey, 1,
                            sent);
      CHECK(rc >= 0, "write %u", sent);
      if (rc == 0) break;
      ++sent;
    }
    imx_qp_poll(tx, imms, 0, &r);
    CHECK(r.n_failed == 0, "tx failed: %s", imx_qp_errors(tx));
    CHECK(imx_qp_poll(rx, imms, DEPTH, &r) == 0, "poll rx");
    CHECK(r.n_failed == 0, "rx failed: %s", imx_qp_errors(rx));
    for (int i = 0; i < r.n_imm; ++i) CHECK(imms[i] == received + i, "order %u", imms[i]);
    received += r.n_imm;
    CHECK(imx_qp_post_recvs(rx, r.n_imm) == r.n_imm, "re-post");
    if (received - acked >= 64) {
      uint64_t pos = received;
      if (imx_qp_write_inline(rx, &pos, 8, credit_info.addr, credit_info.rkey) == 1) acked = received;
    }
    if (now() - t0 > 30) CHECK(0, "timeout at %u/%d frames", received, frames);
  }
  double dt = now() - t0;
  imx_stats s;
  imx_qp_stats_get(rx, &s);
  printf("%d frames in %.3f s (%.0f k/s); rx: %llu imm, %d RECEIVEs posted, %llu credit writes "
         "(%d outstanding)\n",
         frames, dt, frames / dt / 1e3, (unsigned long long)s.imm_completions, s.recvs_outstanding,
         (unsigned long long)s.writes_posted, s.writes_outstanding);
  CHECK(s.recvs_outstanding == DEPTH, "RECEIVEs kept at the depth");

  /* 4. A bad rkey puts the connection in the error state and shows up as a failure. */
  CHECK(imx_qp_write(tx, src_mr, 0, 128, ring_info.addr, ring_info.rkey ^ 0xFFFF, 1, 1) == 1,
        "bad write");
  /* Completions of earlier WRITEs may still come first: wait for the failure itself. */
  int failed = 0;
  deadline = now() + 5;
  while (!failed && now() < deadline) {
    CHECK(imx_qp_poll(tx, imms, 0, &r) == 0, "poll tx");
    failed = r.n_failed;
  }
  CHECK(failed > 0, "bad rkey detected");
  printf("bad rkey: %s\n", imx_qp_errors(tx));

  imx_qp_destroy(rx);
  imx_qp_destroy(tx);
  imx_mr_deregister(ring_mr);
  imx_mr_deregister(src_mr);
  imx_mr_deregister(credit_mr);
  imx_ctx_close(ctx);
  free(ring);
  free(src);
  free(credit_word);
  printf("OK\n");
  return 0;
}
