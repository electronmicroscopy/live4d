"""Browser viewer for the live stream (`python -m realtime_stem.streaming.web <endpoint>`)."""

from realtime_stem.streaming.web.server import (  # noqa: F401
    FrameStore,
    build_parser,
    main,
    make_server,
    subscribe,
)
