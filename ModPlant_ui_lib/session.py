from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ModPlant_fsa_checker_core import (
    STEP_PATTERN,
    apply_cost_duration_corrections,
    check_prediction_against_rules,
    get_last_flow_trace,
)
from .modplant_unsloth_runtime import (
    prepare_runtime as prepare_unsloth_runtime,
    seed_everything as seed_runtime,
)
from .pipeline import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LLM_ATTEMPT_TIMEOUT_S,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_LLM_BASE_TEMPERATURE,
    DEFAULT_LLM_BASE_TOP_P,
    DEFAULT_LLM_MAX_SEQ_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_PERSIST_TEMP_FILES,
    DEFAULT_UI_SEED,
    build_context_from_seed,
    compact_json,
    load_fsa_reference_steps,
    run_fsa_solver_pipeline,
    run_llm_autoregressive,
)

OVERVIEW_MODPLANT_COLUMNS = [
    "Unit",
    "Inputs",
    "Outputs",
    "MaxVolume",
    "Resources",
    "Operation",
    "Param",
    "Cost",
]
OVERVIEW_SCHEDULE_COLUMNS = ["Step", "Type", "Stage", "Start(s)", "Duration(s)"]
OVERVIEW_RULE_COLUMNS = ["Inputs", "Reaction Type", "Reaction Param", "Result"]
STEP_TABLE_COLUMNS = ["Step", "Operation", "Cost", "Duration"]


@dataclass(slots=True)
class PipelineEventCallbacks:
    log: Callable[[str], None] | None = None
    context: Callable[[dict[str, Any]], None] | None = None
    llm_load: Callable[[dict[str, Any]], None] | None = None
    llm_step: Callable[[int, str, float], None] | None = None
    llm_preview: Callable[[str, float], None] | None = None
    llm_done: Callable[[str, float], None] | None = None
    llm_check: Callable[[dict[str, Any]], None] | None = None
    fsa_started: Callable[[], None] | None = None
    fsa_decision_needed: Callable[[dict[str, Any]], bool] | None = None
    fsa_done: Callable[[dict[str, Any]], None] | None = None
    failed: Callable[[str], None] | None = None
    done: Callable[[dict[str, Any]], None] | None = None


def _call_optional(func: Callable[..., Any] | None, *args: Any) -> Any:
    if callable(func):
        return func(*args)
    return None


def prewarm_adapter_runtime(
    project_root: Path,
    adapter_dir: str | Path,
    *,
    seed: int = DEFAULT_UI_SEED,
) -> dict[str, Any]:
    return dict(
        prepare_unsloth_runtime(
            project_root=Path(project_root).resolve(),
            adapter_dir=Path(adapter_dir).expanduser().resolve(),
            default_base_model=DEFAULT_BASE_MODEL,
            max_seq_length=DEFAULT_LLM_MAX_SEQ_LENGTH,
            load_in_4bit=DEFAULT_LOAD_IN_4BIT,
            seed=int(seed),
        )
    )


def format_flow_check_result(comp: dict[str, Any] | None) -> str:
    if not comp:
        return "Flow Simulation Check\n---------------------\nStatus : WAITING"

    warnings = comp.get("warnings") or []
    if comp.get("ok") is True and warnings:
        status = "PASS_WITH_WARNING"
    else:
        status = "PASS" if comp.get("ok") else "FAIL"
    step = comp.get("error_step")
    reason = comp.get("reason", "")
    attempt = comp.get("attempt")
    max_attempts = comp.get("max_attempts")
    temperature = comp.get("temperature")
    top_p = comp.get("top_p")
    timed_out = comp.get("timed_out")
    corrections = comp.get("corrections") or []

    lines = [
        "Flow Simulation Check",
        "---------------------",
        f"Status : {status}",
        f"Attempt: {attempt if attempt is not None else '-'} / {max_attempts if max_attempts is not None else '-'}",
        f"Temp   : {float(temperature):.3f}" if temperature is not None else "Temp   : -",
        f"Top-p  : {float(top_p):.3f}" if top_p is not None else "Top-p  : -",
        f"Timeout: {'Yes' if timed_out else 'No'}" if timed_out is not None else "Timeout: -",
        f"Step   : {step if step is not None else '-'}",
        f"Reason : {reason}",
    ]
    if corrections:
        lines.append("")
        lines.append("Corrections:")
        for i, correction in enumerate(corrections, start=1):
            lines.append(f"{i}. {correction}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for i, warning in enumerate(warnings, start=1):
            lines.append(f"{i}. {warning}")
    return "\n".join(lines)


def _step_line_to_row(line: str) -> list[Any] | None:
    match = STEP_PATTERN.match(line.strip())
    if not match:
        return None
    return [
        match.group("idx"),
        match.group("op"),
        f"{float(match.group('cost')):.3f}",
        f"{float(match.group('dur')):.3f}",
    ]


def _is_connect_row(row: list[Any]) -> bool:
    if len(row) < 2:
        return False
    return str(row[1]).strip().lower().startswith("connect(")


def split_step_lines(lines: list[str]) -> dict[str, list[list[Any]]]:
    connect_rows: list[list[Any]] = []
    process_rows: list[list[Any]] = []
    for line in lines:
        row = _step_line_to_row(line)
        if row is None:
            continue
        if _is_connect_row(row):
            connect_rows.append(row)
        else:
            process_rows.append(row)
    for index, row in enumerate(connect_rows, start=1):
        row[0] = str(index)
    for index, row in enumerate(process_rows, start=1):
        row[0] = str(index)
    return {
        "connect_rows": connect_rows,
        "process_rows": process_rows,
    }


def build_overview_data(payload: dict[str, Any]) -> dict[str, Any]:
    context_obj = payload.get("input_context", {}) or {}
    w_ops = context_obj.get("ModPlant_ops", {}) or {}
    w_ifaces = context_obj.get("ModPlant_interfaces", {}) or {}
    w_res = context_obj.get("ModPlant_resources", {}) or {}
    w_cap = context_obj.get("ModPlant_maximum_volume", {}) or {}

    modplant_rows: list[list[Any]] = []
    for wb in sorted(w_ops.keys()):
        ifaces = w_ifaces.get(wb, [])
        in_cnt = sum(1 for kind, _ in ifaces if str(kind).lower() == "input")
        out_cnt = sum(1 for kind, _ in ifaces if str(kind).lower() == "output")
        max_vol = (w_cap.get(wb) or [""])[0]
        res = w_res.get(wb)
        res_text = f"{res[0]}:{res[1]}" if isinstance(res, list) and len(res) >= 2 else ""
        for op in w_ops.get(wb, []):
            op_name, op_param, op_cost = op[0], op[1], op[2]
            modplant_rows.append([wb, in_cnt, out_cnt, max_vol, res_text, op_name, op_param, op_cost])

    schedule_rows: list[list[Any]] = []
    for index, step in enumerate(payload.get("schedule", []) or [], start=1):
        schedule_rows.append(
            [
                index,
                step.get("type", ""),
                step.get("stage", ""),
                step.get("start_s", ""),
                step.get("duration_s", ""),
            ]
        )

    rule_rows: list[list[Any]] = []
    for rule in payload.get("reaction_rules", []) or []:
        rule_rows.append(
            [
                rule.get("Inputs", ""),
                rule.get("Reaction Type", ""),
                rule.get("Reaction Param", ""),
                rule.get("Result", ""),
            ]
        )

    order = payload.get("order", {}) or {}
    return {
        "modplant_columns": list(OVERVIEW_MODPLANT_COLUMNS),
        "modplant_rows": modplant_rows,
        "schedule_columns": list(OVERVIEW_SCHEDULE_COLUMNS),
        "schedule_rows": schedule_rows,
        "rule_columns": list(OVERVIEW_RULE_COLUMNS),
        "rule_rows": rule_rows,
        "recipe_summary": (
            f"Seed={payload.get('seed')} | Volume={order.get('volume', '-')}"
            f" | XML={payload.get('xml_path', '-')}"
        ),
    }


def run_pipeline_session(
    *,
    project_root: Path,
    seed: int,
    adapter_dir: str | Path,
    persist_temp_files: bool = DEFAULT_PERSIST_TEMP_FILES,
    llm_attempt_timeout_s: int = DEFAULT_LLM_ATTEMPT_TIMEOUT_S,
    callbacks: PipelineEventCallbacks | None = None,
    stop_cb: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    callbacks = callbacks or PipelineEventCallbacks()
    project_root = Path(project_root).resolve()
    adapter_dir = Path(adapter_dir).expanduser().resolve()

    def stopped() -> bool:
        return bool(stop_cb and stop_cb())

    def emit_log(message: str) -> None:
        _call_optional(callbacks.log, message)

    try:
        if stopped():
            return {"status": "stopped", "stage": "initial"}
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

        emit_log("Building context from seed/import...")
        emit_log(f"Persist temp files: {bool(persist_temp_files)}")
        seed_runtime(int(seed))
        display_context = build_context_from_seed(int(seed), persist_temp_files=bool(persist_temp_files))
        context_obj = display_context["input_context"]
        context_compact = compact_json(context_obj)
        _call_optional(
            callbacks.context,
            {
                "context_obj": context_obj,
                "display_context": display_context,
            },
        )

        emit_log("Loading LLM model/runtime...")
        llm_load_info = prewarm_adapter_runtime(project_root, adapter_dir, seed=int(seed))
        if stopped():
            emit_log("[LLM] Model loading interrupted; session exiting.")
            return {"status": "stopped", "stage": "model_load"}
        _call_optional(callbacks.llm_load, llm_load_info)
        emit_log(
            "[LLM] Model load done: "
            f"{float(llm_load_info.get('elapsed_s', 0.0)):.2f}s | "
            f"base={llm_load_info.get('base_model_name')} | "
            f"device={llm_load_info.get('device')} | "
            f"cache={'hit' if llm_load_info.get('runtime_cache_hit') else 'miss'}"
        )

        emit_log(
            "Running LLM autoregressive inference (Unsloth/CUDA)... "
            f"temperature={float(DEFAULT_LLM_BASE_TEMPERATURE):.3f}, "
            f"top_p={float(DEFAULT_LLM_BASE_TOP_P):.3f}, "
            f"timeout={int(llm_attempt_timeout_s)}s"
        )
        llm_step_failed = False
        llm_step_fail_payload: dict[str, Any] | None = None
        correction_notes: list[str] = []
        correction_seen: set[str] = set()

        def record_corrections(notes: list[str]) -> list[str]:
            new_notes: list[str] = []
            for note in notes:
                if note in correction_seen:
                    continue
                correction_seen.add(note)
                correction_notes.append(note)
                new_notes.append(note)
            return new_notes

        def step_validator(step_idx: int, line: str, elapsed: float, generated_steps: list[str]) -> bool:
            nonlocal llm_step_failed, llm_step_fail_payload
            check_start = time.perf_counter()
            partial_text = "\n".join(generated_steps)
            partial_check = check_prediction_against_rules(context_obj, partial_text)
            partial_check["elapsed_s"] = time.perf_counter() - check_start
            partial_check["attempt"] = 1
            partial_check["max_attempts"] = 1
            partial_check["temperature"] = float(DEFAULT_LLM_BASE_TEMPERATURE)
            partial_check["top_p"] = float(DEFAULT_LLM_BASE_TOP_P)
            partial_check["timed_out"] = False
            partial_check["step_by_step"] = True
            partial_check["partial"] = True

            corrected_lines, notes, changed = apply_cost_duration_corrections(
                generated_steps,
                list(partial_check.get("warnings") or []),
            )
            if changed:
                generated_steps[:] = corrected_lines
                _call_optional(callbacks.llm_preview, "\n".join(generated_steps), elapsed)
                for message in record_corrections(notes):
                    emit_log(f"[FlowCheck] Auto-fix: {message}")
            partial_check["corrections"] = list(correction_notes)

            if partial_check.get("ok") is True:
                _call_optional(callbacks.llm_check, partial_check)
                return True

            reason = str(partial_check.get("reason", "")).strip()
            last_is_end = "op: end" in line.lower()
            if (not last_is_end) and reason == "No End step produced.":
                partial_check["ok"] = True
                partial_check["reason"] = "Prefix is feasible so far."
                _call_optional(callbacks.llm_check, partial_check)
                return True

            llm_step_failed = True
            partial_check["auto_fsa"] = True
            llm_step_fail_payload = partial_check
            _call_optional(callbacks.llm_check, partial_check)
            return False

        prediction_text, llm_elapsed_total, runtime_info, timed_out, interrupted = run_llm_autoregressive(
            context_str=context_compact,
            project_root=project_root,
            adapter_dir=adapter_dir,
            max_steps=DEFAULT_MAX_STEPS,
            seed=int(seed),
            temperature=float(DEFAULT_LLM_BASE_TEMPERATURE),
            top_p=float(DEFAULT_LLM_BASE_TOP_P),
            attempt_timeout_s=float(llm_attempt_timeout_s),
            stop_cb=stopped,
            progress_cb=lambda step_idx, line, elapsed: _call_optional(
                callbacks.llm_step,
                step_idx,
                line,
                elapsed,
            ),
            step_validator_cb=step_validator,
        )
        if interrupted and not llm_step_failed and stopped():
            emit_log("[LLM] Inference interrupted; session exiting.")
            return {"status": "stopped", "stage": "llm_inference"}

        runtime_info = dict(runtime_info)
        runtime_info["model_load_elapsed_s"] = float(llm_load_info.get("elapsed_s", 0.0))
        runtime_info["model_load_cache_hit"] = bool(llm_load_info.get("runtime_cache_hit"))
        runtime_info["llm_inference_elapsed_s"] = float(llm_elapsed_total)
        emit_log(
            "[LLM] Runtime ready: "
            f"base={runtime_info.get('base_model_name')} | "
            f"device={runtime_info.get('device')} | "
            f"cache={'hit' if runtime_info.get('runtime_cache_hit') else 'miss'}"
        )
        if timed_out:
            emit_log(f"[LLM] Attempt reached timeout {int(llm_attempt_timeout_s)}s.")
        _call_optional(callbacks.llm_done, prediction_text, llm_elapsed_total)

        if llm_step_failed and llm_step_fail_payload is not None:
            llm_check = llm_step_fail_payload
            llm_check["corrections"] = list(correction_notes)
            for trace_line in get_last_flow_trace():
                emit_log(trace_line)
            emit_log(
                f"[FlowCheck] Result=FAIL | error_step={llm_check.get('error_step')} "
                f"| reason={llm_check.get('reason')}"
            )
        else:
            final_lines = [line.strip() for line in prediction_text.splitlines() if line.strip()]
            check_start = time.perf_counter()
            llm_check = check_prediction_against_rules(context_obj, prediction_text)
            llm_check["elapsed_s"] = time.perf_counter() - check_start
            corrected_lines, notes, changed = apply_cost_duration_corrections(
                final_lines,
                list(llm_check.get("warnings") or []),
            )
            if changed:
                prediction_text = "\n".join(corrected_lines)
                _call_optional(callbacks.llm_preview, prediction_text, llm_elapsed_total)
                for message in record_corrections(notes):
                    emit_log(f"[FlowCheck] Auto-fix: {message}")
                check_start = time.perf_counter()
                llm_check = check_prediction_against_rules(context_obj, prediction_text)
                llm_check["elapsed_s"] = time.perf_counter() - check_start
            llm_check["attempt"] = 1
            llm_check["max_attempts"] = 1
            llm_check["temperature"] = float(DEFAULT_LLM_BASE_TEMPERATURE)
            llm_check["top_p"] = float(DEFAULT_LLM_BASE_TOP_P)
            llm_check["timed_out"] = bool(timed_out)
            llm_check["step_by_step"] = True
            llm_check["auto_fsa"] = bool(llm_check.get("ok") is not True)
            llm_check["corrections"] = list(correction_notes)
            for trace_line in get_last_flow_trace():
                emit_log(trace_line)
            status_text = "PASS" if llm_check.get("ok") else "FAIL"
            emit_log(
                f"[FlowCheck] Result={status_text} | error_step={llm_check.get('error_step')} "
                f"| reason={llm_check.get('reason')}"
            )
            _call_optional(callbacks.llm_check, llm_check)

        if llm_check.get("ok") is True:
            emit_log("LLM state-transition check is feasible; waiting for user decision to run FSA+BFS+OPT...")
            go_fsa = False
            if callbacks.fsa_decision_needed is not None:
                go_fsa = bool(_call_optional(callbacks.fsa_decision_needed, llm_check))
            if stopped():
                emit_log("[LLM] Stopping before FSA by user request.")
                return {"status": "stopped", "stage": "before_fsa"}
            if not go_fsa:
                result = {
                    "status": "ok",
                    "llm_elapsed_s": llm_elapsed_total,
                    "flow_check_elapsed_s": llm_check.get("elapsed_s", 0.0),
                    "fsa_elapsed_s": 0.0,
                    "fsa_skipped": True,
                    "llm_attempts": llm_check.get("attempt"),
                    "runtime": runtime_info,
                }
                _call_optional(callbacks.done, result)
                return result
        else:
            emit_log("LLM step-level flow check failed; auto-running FSA+BFS+OPT for verification.")

        emit_log("Running FSA + solver pipeline...")
        _call_optional(callbacks.fsa_started)
        if stopped():
            emit_log("[FSA] Stopping by user request.")
            return {"status": "stopped", "stage": "before_fsa_run"}

        fsa = run_fsa_solver_pipeline(
            project_root=project_root,
            fsa_python=None,
            seed=int(seed),
            persist_recipe_files=bool(persist_temp_files),
            save_corpus_jsonl=bool(persist_temp_files),
            log_cb=emit_log,
        )

        reference_lines: list[str] = []
        if fsa.get("corpus_path"):
            reference_lines = load_fsa_reference_steps(str(fsa["corpus_path"]))
        if not reference_lines:
            worker_result = fsa.get("worker_result") or {}
            raw_lines = worker_result.get("step_lines") or []
            reference_lines = [str(item).strip() for item in raw_lines if str(item).strip()]

        fsa_payload = {
            **fsa,
            "reference_steps": reference_lines,
        }
        _call_optional(callbacks.fsa_done, fsa_payload)

        result = {
            "status": "ok",
            "llm_elapsed_s": llm_elapsed_total,
            "fsa_elapsed_s": fsa.get("elapsed_s", 0.0),
            "flow_check_elapsed_s": llm_check.get("elapsed_s", 0.0),
            "fsa_skipped": False,
            "llm_attempts": llm_check.get("attempt"),
            "runtime": runtime_info,
        }
        _call_optional(callbacks.done, result)
        return result
    except Exception as exc:
        tb = traceback.format_exc(limit=8)
        message = f"{exc}\n\n{tb}"
        _call_optional(callbacks.failed, message)
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
        }


__all__ = [
    "OVERVIEW_MODPLANT_COLUMNS",
    "OVERVIEW_RULE_COLUMNS",
    "OVERVIEW_SCHEDULE_COLUMNS",
    "PipelineEventCallbacks",
    "STEP_TABLE_COLUMNS",
    "build_overview_data",
    "format_flow_check_result",
    "prewarm_adapter_runtime",
    "run_pipeline_session",
    "split_step_lines",
]
