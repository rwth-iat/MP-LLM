#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .modplant_unsloth_runtime import bootstrap_unsloth
from .runtime_preload import RUNTIME_LIB_NAMES, preload_runtime_libs


def _ensure_qt_runtime_loader(project_root: Path) -> None:
    if sys.platform != "linux":
        return
    if os.environ.get("MODPLANT_QT_RUNTIME_READY") == "1":
        return

    source_dir = Path(sys.prefix).resolve() / "lib"
    runtime_dir = project_root / ".qt_runtime_libs"
    runtime_dir.mkdir(exist_ok=True)

    for lib_name in RUNTIME_LIB_NAMES:
        src = source_dir / lib_name
        if not src.is_file():
            continue
        dst = runtime_dir / lib_name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    env = os.environ.copy()
    env["MODPLANT_QT_RUNTIME_READY"] = "1"
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(runtime_dir) + (os.pathsep + current if current else "")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main(project_root: Path) -> int:
    project_root = Path(project_root).resolve()
    preload_runtime_libs()
    _ensure_qt_runtime_loader(project_root)
    bootstrap_unsloth()

    from PyQt6.QtWidgets import QApplication

    from .window import WabenPlannerWindow

    app = QApplication(sys.argv)
    win = WabenPlannerWindow(project_root=project_root)
    win.show()
    return app.exec()
