"""Holoscan source operator RdmaRxOp: receives imagemux's frames from the ring(s) and emits them
as Start / Batch / End lists per ring, and StemOp, which turns those of one ring into 4D-STEM
tensors with scan positions. The operators that consume them belong to the app, e.g.
WorkerOp, AggregateOp and VirtualDetectorOp in examples/pipeline/operators.py.
"""

import time

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec

from .dispatch import (DeviceRing, Dispatcher, flush_gpudirect_writes, ring_location,
                      set_sync_memops)
from .imagemux import (Compression, CreditTracker, DataType, Frame, FrameKind, FrameReader,
                      RemoteRegion)
from .messages import (IMAGE_BYTES, IMAGE_SHAPE, Batch, End, ScanShape, Segment, Start,
                       parse_start, scan_positions)


# ---------------------------------------------------------------------------------------------
# RdmaRxOp
#
# imagemux receiver: frames arrive by RDMA WRITE_WITH_IMM into a ring, leave as GPU tensors.
# ---------------------------------------------------------------------------------------------

GPU_RING_SETTLE_S = 0.01  # time to wait for a header to show up in a ring in GPU memory
MAX_DEFICIT = 256  # RECEIVEs that may be missing before it is an error; the sender has 1024


class RingPath:
    """One ring, a memory region the sender writes frames into, with everything the receiver keeps
    for it: where the CPU reads headers (``view``), the frame reader, the credits, and the
    dispatcher that gets the images onto the ring's GPU.

    The ring is in GPU memory (a device memory region, GPUDirect RDMA) and its images are
    processed on that GPU: a GPU only ever reads its own ring. The CPU reads the frame headers
    from the ring, which in GPU memory means copying them out.
    """

    def __init__(self, index: int, ring: RemoteRegion, credit: RemoteRegion):
        self.index = index
        self.ring = ring
        self.credit = credit
        self.ring_gpu = ring_location(ring.addr)  # GPU holding the ring
        if self.ring_gpu is None:
            raise ValueError(f"ring {index} is not in GPU memory")
        self.device = self.ring_gpu
        # The NIC's writes into GPU memory may become visible a little after the completion.
        self.reader = FrameReader(ring.length, settle_s=GPU_RING_SETTLE_S)
        self.view = DeviceRing(ring.addr, ring.length, self.ring_gpu)
        status = set_sync_memops(ring.addr, self.ring_gpu)
        if status != 0:
            print(f"cuPointerSetAttribute(SYNC_MEMOPS) on ring {index} returned {status}",
                  flush=True)
        self.credits = CreditTracker()  # ring positions whose GPU reads have finished
        self.credit_sent = 0
        self.dispatcher = None  # created in start(), on the thread that runs the operator

    def start(self, reduce: bool):
        self.dispatcher = Dispatcher(self.device, reduce=reduce)

    @property
    def held(self) -> int:
        """Ring bytes read but not yet returned to the sender as credit."""
        return self.reader.position - self.credit_sent

    @property
    def where(self) -> str:
        return f"memory of GPU {self.ring_gpu} (GPUDirect RDMA)"


class RdmaRxOp(Operator):
    """Source: reads the frames imagemux writes into the ring(s) and emits, on ``output{i}`` of
    each ring ``i``, a list in arrival order of ``Batch`` (images of one series on that ring's
    GPU, as pixel sums or as images, with host image ids restarting at 0 per series), ``Start``
    and ``End`` (with the image count); ``Start`` and ``End`` go to every output.

    There is one ``RingPath`` per ring (currently one). ``conn`` is the connected
    ``imxrdma.QueuePair``; it is only used from this operator's compute.
    """

    def __init__(self, fragment, *args, conn, ring: RemoteRegion, credit: RemoteRegion,
                 reduce: bool = True, max_batch: int = 512,
                 receive_only: bool = False, **kwargs):
        self.conn = conn
        self.imms = np.empty(max_batch, dtype=np.uint32)  # filled by conn.poll
        self.reduce = reduce
        self.receive_only = receive_only  # count images and return the credit, no GPU work
        self.max_batch = max_batch
        self.paths = [RingPath(0, ring, credit)]
        what = ("receive only: frames are counted and their credit returned at once, nothing "
                "is processed" if receive_only
                else "pixel sums are computed straight from the ring" if reduce
                else "images are copied to GPU memory")
        for path in self.paths:
            print(f"ring {path.index} of {path.ring.length >> 20} MiB in {path.where}; {what}",
                  flush=True)
        self.series = None  # Series between its start and end frame; None outside or if rejected
        self.images = 0  # images of the series received, or dropped if it has no Series
        self.next_id, self.missing, self.gaps_shown = None, 0, 0  # image id continuity check
        self.peak_fill = 0.0  # largest share of a ring held (read, credit not returned yet)
        self.lost = False  # the connection failed; the app is being stopped
        self.deficit = 0  # RECEIVEs that could not be re-posted yet
        super().__init__(fragment, *args, **kwargs)

    def path_of(self, imm: int) -> RingPath:
        """The ring that the write announced by ``imm`` went to. There is only one for now."""
        return self.paths[0]

    def setup(self, spec: OperatorSpec):
        for path in self.paths:
            spec.output(f"output{path.index}")

    def initialize(self):
        super().initialize()
        for path in self.paths:
            path.start(self.reduce)

    def start_series(self, frame: Frame, path: RingPath):
        """Returns the Start marker of the series, or None if it was rejected."""
        header = frame.header
        self.images = 0
        self.next_id, self.missing, self.gaps_shown = None, 0, 0
        self.peak_fill = 0.0
        self.first_image = None  # when the first image frame arrived (START comes earlier)
        try:
            data = path.view[frame.data_offset:frame.data_offset + header.data_size]
            self.series = parse_start(data)
        except ValueError as e:
            self.series = None
            print(f"series {header.series_id} rejected: {e}", flush=True)
            return None
        rate = (f", announced rate {1e-3 / self.series.frame_time:.1f} kHz "
                f"({self.series.frame_time * 1e6:.1f} us per image)"
                if self.series.frame_time else "")
        print(f"series {header.series_id} START: ({self.series.expected_images or '?'} images "
              f"announced{rate})", flush=True)
        return Start(self.series)

    def end_series(self, frame: Frame):
        """Returns the End marker of the series, or None if it was not being processed."""
        series, self.series = self.series, None
        if series is None:
            print(f"series {frame.header.series_id} END: {self.images} images dropped", flush=True)
            return None
        lost = f", {self.missing} missing (gaps in the image ids)" if self.missing else ""
        took = time.perf_counter() - (self.first_image or time.perf_counter())
        rate = f" in {took:.2f} s ({self.images / took / 1e3:.1f} kHz)" if took > 0 else ""
        print(f"series {series.id} END: {self.images} images{lost}{rate}; ring held at most "
              f"{100 * self.peak_fill:.0f}%", flush=True)
        return End(series, self.images)

    def frame_images(self, frame: Frame, path: RingPath):
        """(addresses, ids) of the images of an image frame, or None if it carries none to use."""
        header = frame.header
        image = header.image
        if image.channel != 0:
            return None
        if self.series is None:  # rejected, or the receiver joined in the middle of a series
            self.images += image.batch_size
            return None
        if (image.compression != Compression.NONE or image.dtype != DataType.UINT16
                or (image.dim_y, image.dim_x) != IMAGE_SHAPE
                or header.data_size != image.batch_size * IMAGE_BYTES):
            raise RuntimeError(
                f"frame {header.sequence}: unsupported image data {image.dim_x}x{image.dim_y} "
                f"{image.dtype.name} {image.compression.name}, batch {image.batch_size}, "
                f"{header.data_size} bytes; expected uncompressed {IMAGE_SHAPE} uint16")
        if self.first_image is None:
            self.first_image = time.perf_counter()
        if self.next_id is not None and image.image_id != self.next_id:
            gap = image.image_id - self.next_id  # negative: ids went backwards
            self.missing += max(gap, 0)
            if self.gaps_shown < 5:
                self.gaps_shown += 1
                print(f"series {self.series.id}: image id {image.image_id} follows "
                      f"{self.next_id - 1} ({gap} skipped) at frame {header.sequence}, ring "
                      f"{100 * path.held // path.ring.length}% "
                      f"held", flush=True)
        self.next_id = image.image_id + image.batch_size
        self.images += image.batch_size
        if self.receive_only:
            return None
        steps = np.arange(image.batch_size, dtype=np.uint64)
        addrs = np.uint64(path.ring.addr + frame.data_offset) + steps * np.uint64(IMAGE_BYTES)
        return addrs, np.uint64(image.image_id) + steps

    def repost_recvs(self, n: int):
        """Re-arm ``n`` zero-length RECEIVEs, plus any earlier shortfall. They do not protect the
        ring (the credit does), but each WRITE_WITH_IMM consumes one, so the sender stalls if we
        run out. A shortfall (the receive queue refused some) is carried over and retried on the
        next tick; a large one is an error."""
        want = n + self.deficit
        if want == 0:
            return
        posted = self.conn.post_recvs(want)
        if posted < want and self.deficit == 0:
            print(f"re-posted {posted}/{want} RECEIVEs; carrying the shortfall over "
                  f"({self.conn.stats})", flush=True)
        self.deficit = want - posted
        if self.deficit > MAX_DEFICIT:
            raise RuntimeError(f"{self.deficit} RECEIVEs could not be re-posted; the sender "
                               f"would run out of them")

    def connection_lost(self, failed: int):
        """A failed completion means the queue pair is in the error state (a flush of every posted
        RECEIVE follows the failure that caused it). Stop the app instead of raising, with what
        we know on our side and the first errors the queue pair saw."""
        self.lost = True
        rings = "; ".join(f"ring {p.index}: {p.reader.sequence} frame(s), position "
                          f"{p.reader.position}, credit sent {p.credit_sent}" for p in self.paths)
        print(f"connection lost: {failed} failed RDMA completion(s) on qp "
              f"{self.conn.endpoint.qp_num} ({rings}; series "
              f"{self.series.id if self.series else None}, {self.images} images): "
              f"{self.conn.errors or 'no error recorded'}. If it is only a flush, the QP "
              f"was put in the error state without a completion, typically by a remote access "
              f"error on one of imagemux's plain RDMA writes (QP access flags, ring rkey or "
              f"address range)", flush=True)
        self.fragment.stop_execution()

    def send_credit(self, path: RingPath):
        """Return the ring position up to which the GPUs are done reading to imagemux. It is
        cumulative, so a credit that cannot be posted now is simply retried on the next tick."""
        position = path.credits.poll()
        if position == path.credit_sent:
            return
        if self.conn.write_u64(position, path.credit.addr, path.credit.rkey):
            path.credit_sent = position
        else:  # the send queue is full
            print(f"credit write not posted (ring {path.index}, position={position})", flush=True)

    def compute(self, op_input, op_output, context):
        self.receive(op_output)

    def receive(self, op_output) -> int:
        """One poll of the receive loop; returns the number of completions it found.
        Drains up to max_batch completions without blocking. Each one announces a frame written
        into the ring; its immediate is the frame's sequence number. Completed credit writes are
        reaped in the same poll."""
        if self.lost:
            return 0
        if self.deficit:
            self.repost_recvs(0)
        k, _, failed = self.conn.poll(self.imms)
        imms = self.imms[:k]
        if failed:
            # A failed completion (e.g. bad rkey) leaves the QP in error: stop.
            self.connection_lost(failed)
            return 0
        if k == 0:
            for path in self.paths:
                self.send_credit(path)  # retry a credit that could not be posted
            return 0
        # Re-arm the zero-length RECEIVEs first so that their number stays constant. They do not
        # protect the ring; the credit does.
        self.repost_recvs(k)
        for path in self.paths:
            flush_gpudirect_writes(path.ring_gpu)  # the NIC's writes must be visible

        # In arrival order: [path, series, [addresses], [ids]] runs of images of one series from
        # one ring, and End markers between them.
        sequence = []
        for imm in imms:
            path = self.path_of(int(imm))
            frame = path.reader.next(path.view, int(imm) & 0xFFFFFFFF)
            if frame is None:
                continue
            kind = frame.header.kind
            if kind == FrameKind.START:
                if (start := self.start_series(frame, path)) is not None:
                    sequence.append(start)
            elif kind == FrameKind.END:
                if (end := self.end_series(frame)) is not None:
                    sequence.append(end)
            elif (images := self.frame_images(frame, path)) is not None:
                if not sequence or not isinstance(sequence[-1], list) \
                        or sequence[-1][0] is not path or sequence[-1][1] is not self.series:
                    sequence.append([path, self.series, [], []])
                sequence[-1][2].append(images[0])
                sequence[-1][3].append(images[1])

        self.peak_fill = max([self.peak_fill] + [p.held / p.ring.length for p in self.paths])

        if self.receive_only:
            for path in self.paths:
                path.credits.add(path.reader.position)
                self.send_credit(path)
            return k

        # The kernels read the images straight out of the ring and are not waited for: the ring
        # positions are only credited back once their events have happened (CreditTracker).
        out = {p.index: [] for p in self.paths}  # ring -> [Batch | End]
        launched = {p.index: ([], []) for p in self.paths}  # ring -> (events, keep)
        for event in sequence:
            if isinstance(event, (Start, End)):
                for items in out.values():  # every output starts and ends the series, even with
                    items.append(event)  # no images
                continue
            path, series, addrs, ids = event
            dispatched = path.dispatcher.dispatch(np.concatenate(addrs), np.concatenate(ids))
            events, keep = launched[path.index]
            events.append(dispatched.event)
            keep += dispatched.keep
            data = {"sums": dispatched.payload} if self.reduce else {"images": dispatched.payload}
            out[path.index].append(Batch(series, dispatched.ids, dispatched.event, **data))
        for path in self.paths:
            events, keep = launched[path.index]
            path.credits.add(path.reader.position, events, keep)
            self.send_credit(path)
        for index, items in out.items():
            if items:
                op_output.emit(items, f"output{index}")
        return k


# ---------------------------------------------------------------------------------------------
# StemOp
#
# 4D-STEM tensors: the images of one ring with their scan positions, one emission per batch.
# ---------------------------------------------------------------------------------------------

class StemOp(Operator):
    """Consumes the lists of ``Start``, ``Batch`` and ``End`` items of one ring (an ``RdmaRxOp``
    with ``reduce=False``) on that ring's GPU ``device`` and emits, for each list, one tensor map
    on ``frames`` and the order of things on ``index``.

    ``frames`` (only if there are images): ``diffraction`` (k, 96, 96) uint16, ``positions``
    (k, 2) int32 ``[row, column]`` in the raster ``scan``, and ``ids`` (k,) int64 image ids, all
    CuPy arrays on ``device``. They are complete when emitted, so they can be read on any stream
    (and handed to e.g. PyTorch with DLPack). Images outside the scan are left out.

    ``index`` (always): a list in arrival order of ``Start``, ``Segment`` (rows of ``frames`` that
    are images of one series) and ``End``. A port carries one message per tick, so this is how a
    consumer tells the series in one ``frames`` apart and orders them with their start and end.
    """

    def __init__(self, fragment, *args, device: int, scan: ScanShape, **kwargs):
        self.device = cp.cuda.Device(device)
        self.scan = scan
        self.warned = None  # series that was told about images outside the scan
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("frames")
        spec.output("index")

    def initialize(self):
        super().initialize()
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)

    def _positioned(self, batch: Batch):
        """(images, positions, ids) of the images of ``batch`` that are inside the scan."""
        if batch.images is None:
            raise RuntimeError("StemOp needs the images: an RdmaRxOp with reduce=False "
                               "(receiver.keep_images: true)")
        self.stream.wait_event(batch.ready)  # the images are copied on the dispatcher's stream
        ids = np.asarray(batch.ids, dtype=np.int64)
        positions, valid = scan_positions(ids, self.scan)
        images = batch.images
        if not valid.all():
            if self.warned != batch.series.id:
                self.warned = batch.series.id
                print(f"series {batch.series.id}: images beyond the {self.scan.width}x"
                      f"{self.scan.height} scan are left out (first id {ids[~valid][0]})",
                      flush=True)
            images = images[cp.asarray(np.flatnonzero(valid))]
            ids = ids[valid]
        return images, positions, ids

    def compute(self, op_input, op_output, context):
        index, images, positions, ids = [], [], [], []
        rows = 0
        with self.device, self.stream:
            for item in op_input.receive("input"):
                if not isinstance(item, Batch):
                    index.append(item)
                    continue
                batch_images, batch_positions, batch_ids = self._positioned(item)
                if not len(batch_ids):
                    continue
                index.append(Segment(item.series, rows, rows + len(batch_ids)))
                rows += len(batch_ids)
                images.append(batch_images)
                positions.append(batch_positions)
                ids.append(batch_ids)
            if rows:
                frames = {
                    "diffraction": images[0] if len(images) == 1 else cp.concatenate(images),
                    "positions": cp.asarray(np.concatenate(positions)),
                    "ids": cp.asarray(np.concatenate(ids)),
                }
                # Complete before it leaves: consumers may read it on any stream.
                self.stream.synchronize()
                op_output.emit(frames, "frames")
        op_output.emit(index, "index")
