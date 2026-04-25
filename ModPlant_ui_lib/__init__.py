from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "main",
    "WabenPlannerWindow",
    "PipelineWorker",
    "PipelineEventCallbacks",
    "build_context_from_seed",
    "format_flow_check_result",
    "prewarm_adapter_runtime",
    "run_fsa_solver_pipeline",
    "run_llm_autoregressive",
    "run_pipeline_session",
    "split_step_lines",
]

_EXPORTS = {
    "main": (".entry", "main"),
    "WabenPlannerWindow": (".window", "WabenPlannerWindow"),
    "PipelineWorker": (".workers", "PipelineWorker"),
    "PipelineEventCallbacks": (".session", "PipelineEventCallbacks"),
    "build_context_from_seed": (".pipeline", "build_context_from_seed"),
    "format_flow_check_result": (".session", "format_flow_check_result"),
    "prewarm_adapter_runtime": (".session", "prewarm_adapter_runtime"),
    "run_fsa_solver_pipeline": (".pipeline", "run_fsa_solver_pipeline"),
    "run_llm_autoregressive": (".pipeline", "run_llm_autoregressive"),
    "run_pipeline_session": (".session", "run_pipeline_session"),
    "split_step_lines": (".session", "split_step_lines"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
