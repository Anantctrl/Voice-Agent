"""Central logging setup (moved out of main so runners share one policy)."""
from __future__ import annotations

import logging

_NOISY = ("websockets", "httpcore", "httpx", "groq")


def configure_logging(debug: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Transport chatter (every mic frame / HTTP chunk) buries our lines even
    # in --debug; provider failures still surface via voice_agent loggers.
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
