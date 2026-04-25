#!/usr/bin/env python3
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from ModPlant_ui_lib.entry import main as _package_main

project_root = Path(__file__).resolve().parent

__all__ = [
    "main",
    "WabenPlannerWindow",
    "PipelineWorker",
    "build_context_from_seed",
    "run_fsa_solver_pipeline",
    "run_llm_autoregressive",
]

_EXPORTS = {
    "WabenPlannerWindow": ("ModPlant_ui_lib", "WabenPlannerWindow"),
    "PipelineWorker": ("ModPlant_ui_lib", "PipelineWorker"),
    "build_context_from_seed": ("ModPlant_ui_lib", "build_context_from_seed"),
    "run_fsa_solver_pipeline": ("ModPlant_ui_lib", "run_fsa_solver_pipeline"),
    "run_llm_autoregressive": ("ModPlant_ui_lib", "run_llm_autoregressive"),
}


def main(project_root: Path = project_root) -> int:
    return _package_main(project_root=project_root)


def __getattr__(name: str) -> Any:
    if name == "main":
        return main
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if __name__ == "__main__":
    raise SystemExit(main(project_root=project_root))
