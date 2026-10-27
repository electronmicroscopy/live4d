"""Saved live scans by name, loaded onto the GPU on demand."""

from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_FOLDER = os.environ.get("LIVE4D_FOLDER", "live_scans")


class LiveScans:
    """The scans a `LiveFrames` receiver saved, by name, as ``Dataset4dstemGPU``.

    The receiver saves each completed scan as ``<folder>/<date>/series_297_512x512_96px.qem``
    (the exact ANS bytes). It does not need to run in this process:

        scans = LiveScans("/data/live")  # today's scans, newest last
        data = scans["series_297"]       # by the detector's series name (or the full file stem)
        data = scans.latest()
        series = scans.last(7)
    """

    def __init__(self, folder=DEFAULT_FOLDER, date=None, device="cuda:0"):
        self.folder = Path(folder).expanduser() / (date or time.strftime("%Y-%m-%d"))
        self.device = device
        self._loaded = {}

    def paths(self) -> dict:
        """{name: path}, oldest first; a name is the file stem (series_297_512x512_96px)."""
        files = sorted(self.folder.glob("*.qem"), key=lambda path: path.stat().st_mtime)
        return {path.stem: path for path in files}

    def names(self) -> list:
        return list(self.paths())

    def __len__(self) -> int:
        return len(self.paths())

    def __getitem__(self, name):
        """A scan by name (series_297, or the whole stem) or by index (scans[-1] = newest)."""
        from quantem.gpu import io
        paths = self.paths()
        if isinstance(name, int):
            name = list(paths)[name]
        matches = [stem for stem in paths if stem == name or stem.startswith(f"{name}_")]
        if not matches:
            raise KeyError(f"no saved scan {name!r} in {self.folder}; have: {', '.join(paths) or 'none'}")
        stem = matches[-1]
        if stem not in self._loaded:
            self._loaded[stem] = io.load(paths[stem], device=self.device)
        return self._loaded[stem]

    def latest(self):
        return self[-1]

    def last(self, count: int) -> list:
        return [self[name] for name in self.names()[-count:]]

    def __repr__(self) -> str:
        names = self.names()
        shown = "\n".join(f"  {name}" for name in names[-12:])
        more = f"\n  ... {len(names) - 12} earlier" if len(names) > 12 else ""
        return f"LiveScans({self.folder}, {len(names)} scans, newest last){more}\n{shown}"
