from __future__ import annotations

import ctypes
import sys
from pathlib import Path


RUNTIME_LIB_NAMES = (
    "libXau.so.6",
    "libXdmcp.so.6",
    "libxcb.so.1",
    "libxcb-util.so.1",
    "libxcb-image.so.0",
    "libxcb-keysyms.so.1",
    "libxcb-render-util.so.0",
    "libxcb-icccm.so.4",
    "libxcb-ewmh.so.2",
    "libxcb-cursor.so.0",
    "libxcb-randr.so.0",
    "libxcb-shm.so.0",
    "libxcb-sync.so.1",
    "libxcb-xfixes.so.0",
    "libxcb-render.so.0",
    "libxcb-shape.so.0",
    "libxcb-xkb.so.1",
    "libxkbcommon.so.0",
    "libxkbcommon-x11.so.0",
    "libX11.so.6",
    "libX11-xcb.so.1",
    "libXext.so.6",
    "libXrender.so.1",
    "libXfixes.so.3",
    "libXi.so.6",
    "libXrandr.so.2",
    "libXinerama.so.1",
    "libICE.so.6",
    "libSM.so.6",
    "libzstd.so.1",
    "libGLdispatch.so.0",
    "libOpenGL.so.0",
    "libGLX.so.0",
    "libGL.so.1",
    "libEGL.so.1",
)


def preload_runtime_libs(lib_dir: Path | None = None) -> None:
    lib_dir = (lib_dir or (Path(sys.prefix).resolve() / "lib")).resolve()
    if not lib_dir.is_dir():
        return

    for lib_name in RUNTIME_LIB_NAMES:
        candidate = lib_dir / lib_name
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError:
            pass
