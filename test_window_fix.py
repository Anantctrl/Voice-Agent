"""Workaround runner: disable numba JIT before importing librosa.

Windows App Control blocks llvmlite.dll (WinError 4551), which is loaded when
numba/librosa causes JIT. Setting NUMBA_DISABLE_JIT=1 before import lets
librosa load and this script then runs the normal pipeline offline.
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import sys

sys.path.insert(0, r"D:\Project\voiceai\src")

from audiopreprocessing.main import main

if __name__ == "__main__":
    raise SystemExit(main())
