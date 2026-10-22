"""The settings of app.py: the sections of its Holoscan YAML file (pipeline.yaml) laid over the
defaults below. No holoscan imports, so it can be tested on its own."""

REQUIRED = object()

DEFAULTS = {
    "connection": {
        "remote_address": REQUIRED,  # imagemux bstream RDMA handshake address, e.g. 10.0.0.1:31003
        "local_address": REQUIRED,  # address of the RoCE interface to use
        "device": "",  # RDMA device; "" = the one holding local_address
        "ring_gpu": 0,  # GPU holding the ring
        "ring_mib": 64,
        "timeout": 10,  # seconds for the TCP connection to imagemux
    },
    "execute": {
        "seconds": 10.0,  # 0 = until Ctrl+C
        "threads": 3,  # source, worker and aggregator run concurrently
    },
    "receiver": {  # RdmaRxOp
        "receive_only": False,  # count images and return the credit at once, no GPU work
        "keep_images": False,  # copy the images to GPU memory; False reduces them to pixel sums from the ring
        "max_batch": 512,  # most completions (frames) taken from the receive queue per poll
    },
    "aggregate": {  # AggregateOp
        "series_width": 0,  # images per series row; 0 = a single row
    },
}


def resolve(sections: dict) -> dict:
    """``sections`` ({section name: {key: value}}, as read from the YAML) over DEFAULTS. Unknown
    sections or keys and missing required keys raise ValueError."""
    unknown = sorted(set(sections) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config section(s): {', '.join(unknown)}")
    settings = {}
    for name, defaults in DEFAULTS.items():
        given = sections.get(name) or {}
        bad = sorted(set(given) - set(defaults))
        if bad:
            raise ValueError(f"unknown key(s) in '{name}': {', '.join(bad)}")
        merged = {**defaults, **given}
        missing = [k for k, v in merged.items() if v is REQUIRED]
        if missing:
            raise ValueError(f"'{name}' needs: {', '.join(missing)}")
        settings[name] = merged
    return settings
