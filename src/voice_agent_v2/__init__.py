"""Backwards-compatibility shim: the engine moved to :mod:`voice_agent`.

Everything here aliases the new package, so existing imports, scripts, and
checkouts keep working during the transition. New code should import from
``voice_agent`` directly.
"""
from __future__ import annotations

import contextlib
import pkgutil as _pkgutil
import sys as _sys

import voice_agent as _target

_new_prefix = _target.__name__ + "."
_old_prefix = __name__ + "."

for _found in _pkgutil.walk_packages(_target.__path__, _new_prefix):
    _alias = _old_prefix + _found.name[len(_new_prefix):]
    with contextlib.suppress(Exception):
        _sys.modules.setdefault(_alias, __import__(_found.name, fromlist=["*"]))

from voice_agent import *  # noqa: F403
