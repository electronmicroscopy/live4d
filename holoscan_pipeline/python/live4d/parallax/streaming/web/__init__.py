"""Browser viewer for the live stream (`python -m live4d.parallax.streaming.web <endpoint>`)."""

from live4d.parallax.streaming.web.server import (  # noqa: F401
    FrameStore,
    build_parser,
    main,
    make_server,
    subscribe,
)
