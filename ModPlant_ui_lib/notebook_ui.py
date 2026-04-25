from __future__ import annotations

import html
import random
import threading
import traceback
import time
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import pandas as pd

from .pipeline import (
    DEFAULT_ADAPTER_REL_PATH,
    DEFAULT_LLM_ATTEMPT_TIMEOUT_S,
    DEFAULT_PERSIST_TEMP_FILES,
    DEFAULT_UI_SEED,
    build_context_from_seed,
)
from .session import (
    PipelineEventCallbacks,
    STEP_TABLE_COLUMNS,
    build_overview_data,
    format_flow_check_result,
    prewarm_adapter_runtime,
    run_pipeline_session,
    split_step_lines,
)


class ModPlantNotebookUI:
    def __init__(self, project_root: Path, auto_prewarm_default_adapter: bool = True):
        self.project_root = Path(project_root).resolve()
        self.auto_prewarm_default_adapter = bool(auto_prewarm_default_adapter)
        self.default_adapter_dir = str((self.project_root / DEFAULT_ADAPTER_REL_PATH).resolve())
        self._built = False
        self._pipeline_thread: threading.Thread | None = None
        self._prewarm_thread: threading.Thread | None = None
        self._timer_thread: threading.Thread | None = None
        self._timer_stop_event = threading.Event()
        self._decision_event = threading.Event()
        self._decision_value: bool | None = None
        self.log_lines: list[str] = []
        self.context_payload: dict[str, Any] | None = None
        self.llm_lines: list[str] = []
        self.fsa_lines: list[str] = []
        self.pipeline_result: dict[str, Any] | None = None
        self.run_start = 0.0
        self.total_pause_started_at = 0.0
        self.total_paused_elapsed_s = 0.0
        self.llm_running = False
        self.llm_phase = "idle"
        self.llm_load_start_ts = 0.0
        self.llm_load_elapsed_s = 0.0
        self.llm_inference_elapsed_s = 0.0
        self.fsa_running = False
        self.llm_start_ts = 0.0
        self.fsa_start_ts = 0.0

    def _total_elapsed_s(self) -> float:
        if self.run_start <= 0:
            return 0.0
        paused = self.total_paused_elapsed_s
        if self.total_pause_started_at > 0:
            paused += time.perf_counter() - self.total_pause_started_at
        return max(0.0, time.perf_counter() - self.run_start - paused)

    def _total_label_text(self) -> str:
        return f"Total: {self._total_elapsed_s():.2f}s"

    def _pause_total_timer(self) -> None:
        if self.run_start > 0 and self.total_pause_started_at <= 0:
            self.total_pause_started_at = time.perf_counter()

    def _resume_total_timer(self) -> None:
        if self.total_pause_started_at > 0:
            self.total_paused_elapsed_s += time.perf_counter() - self.total_pause_started_at
            self.total_pause_started_at = 0.0

    def build(self) -> "ModPlantNotebookUI":
        if self._built:
            return self
        self._build_widgets()
        self._wire_events()
        self._build_sections()
        self._reset_context_outputs()
        self._reset_pipeline_outputs()
        self._set_prewarm_status("Model prewarm: idle", tone="idle")
        if self.auto_prewarm_default_adapter:
            self._start_prewarm(auto=True)
        self._built = True
        return self

    def _build_widgets(self) -> None:
        self.adapter_input = widgets.Text(
            value=self.default_adapter_dir,
            description="Adapter:",
            layout=widgets.Layout(width="100%"),
        )
        self.browse_button = widgets.Button(description="Browse Adapter", button_style="")
        self.prewarm_button = widgets.Button(description="Prewarm Model", button_style="info")
        self.prewarm_status_html = widgets.HTML()

        self.persist_files_checkbox = widgets.Checkbox(
            value=DEFAULT_PERSIST_TEMP_FILES,
            description="Persist temporary artifacts",
            indent=False,
        )
        self.timeout_input = widgets.BoundedIntText(
            value=DEFAULT_LLM_ATTEMPT_TIMEOUT_S,
            min=10,
            max=3600,
            description="Timeout (s):",
            layout=widgets.Layout(width="260px"),
        )

        self.seed_input = widgets.BoundedIntText(
            value=DEFAULT_UI_SEED,
            min=1,
            max=10_000_000,
            description="Seed:",
            layout=widgets.Layout(width="240px"),
        )
        self.random_seed_button = widgets.Button(description="Random Seed")
        self.generate_button = widgets.Button(description="Generate From Seed")
        self.context_status_html = widgets.HTML()
        self.recipe_summary_html = widgets.HTML()
        self.modplant_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.schedule_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.rules_output = widgets.HTML(layout=widgets.Layout(width="100%"))

        self.start_button = widgets.Button(description="Start Calculation", button_style="success")
        self.pipeline_status_html = widgets.HTML()
        self.total_time_html = widgets.HTML()
        self.llm_time_html = widgets.HTML()
        self.pre_time_html = widgets.HTML()
        self.fsa_time_html = widgets.HTML()
        self.llm_model_time_html = widgets.HTML()
        self.llm_inference_time_html = widgets.HTML()
        self.llm_connect_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.llm_process_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.llm_check_status_html = widgets.HTML()
        self.llm_check_text = widgets.Textarea(layout=widgets.Layout(width="100%", height="180px"))
        self.fsa_status_html = widgets.HTML()
        self.fsa_connect_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.fsa_process_output = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.log_text = widgets.Textarea(layout=widgets.Layout(width="100%", height="300px"))

        self.decision_status_html = widgets.HTML(value="<em>No decision is pending.</em>")
        self.decision_yes_button = widgets.Button(description="Run FSA+BFS+OPT", button_style="warning")
        self.decision_no_button = widgets.Button(description="Skip FSA+BFS+OPT", button_style="")
        self.decision_button_row = widgets.HBox([self.decision_yes_button, self.decision_no_button])
        self.decision_button_row.layout.display = "none"

    def _wire_events(self) -> None:
        self.browse_button.on_click(self._on_browse_adapter)
        self.prewarm_button.on_click(self._on_prewarm_clicked)
        self.random_seed_button.on_click(self._on_random_seed)
        self.generate_button.on_click(self._on_generate_context)
        self.start_button.on_click(self._on_start_pipeline)
        self.decision_yes_button.on_click(self._on_decision_yes)
        self.decision_no_button.on_click(self._on_decision_no)

    def _build_sections(self) -> None:
        self.settings_section = widgets.VBox(
            [
                self._section_html("Settings", "Configure the adapter, persistence mode, and timeout before running anything."),
                widgets.HBox([self.adapter_input]),
                widgets.HBox([self.browse_button, self.prewarm_button]),
                self.prewarm_status_html,
                self.persist_files_checkbox,
                self.timeout_input,
            ]
        )
        self.context_section = widgets.VBox(
            [
                self._section_html("Seed And Context", "Generate the same ModPlant configuration, recipe schedule, and reaction rules that the Qt home page shows."),
                widgets.HBox([self.seed_input, self.random_seed_button, self.generate_button]),
                self.context_status_html,
                self.recipe_summary_html,
                self._subsection_html("ModPlant Configuration"),
                self.modplant_output,
                self._subsection_html("Schedule"),
                self.schedule_output,
                self._subsection_html("Reaction Rules"),
                self.rules_output,
            ]
        )
        self.pipeline_section = widgets.VBox(
            [
                self._section_html("Pipeline Execution", "Run the full Seed -> LLM -> FSA Checker -> FSA+BFS+OPT flow in a background thread and watch the outputs update in place."),
                widgets.HBox([self.start_button]),
                self.pipeline_status_html,
                widgets.HBox([self.total_time_html, self.llm_time_html, self.pre_time_html, self.fsa_time_html]),
                self._subsection_html("LLM Runtime"),
                widgets.HBox([self.llm_model_time_html, self.llm_inference_time_html]),
                self._subsection_html("LLM Output - Connect"),
                self.llm_connect_output,
                self._subsection_html("LLM Output - Process"),
                self.llm_process_output,
                self._subsection_html("Check Result"),
                self.llm_check_status_html,
                self.llm_check_text,
                self._subsection_html("Decision"),
                self.decision_status_html,
                self.decision_button_row,
                self._subsection_html("FSA+BFS+OPT Status"),
                self.fsa_status_html,
                self._subsection_html("FSA+BFS+OPT Reference - Connect"),
                self.fsa_connect_output,
                self._subsection_html("FSA+BFS+OPT Reference - Process"),
                self.fsa_process_output,
                self._subsection_html("Runtime Log"),
                self.log_text,
            ]
        )
        self.decision_section = widgets.VBox([])

    def _section_html(self, title: str, body: str) -> widgets.HTML:
        return widgets.HTML(
            value=(
                f"<h2 style='margin:0 0 0.35rem 0'>{html.escape(title)}</h2>"
                f"<p style='margin:0 0 0.8rem 0'>{html.escape(body)}</p>"
            )
        )

    def _subsection_html(self, title: str) -> widgets.HTML:
        return widgets.HTML(value=f"<h3 style='margin:0.85rem 0 0.35rem 0'>{html.escape(title)}</h3>")

    def _callout(self, title: str, body: str, color: str) -> str:
        return (
            f"<div style='border-left:4px solid {color};padding:0.55rem 0.8rem;"
            f"background:#f6f8fa;margin:0.35rem 0'>"
            f"<strong>{html.escape(title)}</strong><br>{html.escape(body)}"
            f"</div>"
        )

    def _set_prewarm_status(self, message: str, tone: str = "idle") -> None:
        color = {
            "idle": "#6e7781",
            "running": "#1f6feb",
            "ready": "#238636",
            "failed": "#cf222e",
        }.get(tone, "#6e7781")
        self.prewarm_status_html.value = self._callout("Model prewarm", message, color)

    def _set_pipeline_status(self, title: str, message: str, color: str = "#1f6feb") -> None:
        self.pipeline_status_html.value = self._callout(title, message, color)

    def _set_decision_status(self, title: str, message: str, color: str) -> None:
        self.decision_status_html.value = self._callout(title, message, color)

    def _set_check_status(self, label: str, color: str) -> None:
        badge = (
            f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
            f"background:{color};color:white;font-weight:600'>{html.escape(label)}</span>"
        )
        self.llm_check_status_html.value = badge

    def _set_overview_status(self, message: str, tone: str = "#1f6feb") -> None:
        self.context_status_html.value = self._callout("Seed context", message, tone)

    def _set_runtime_labels(self, *, total: str, llm: str, pre: str, fsa: str) -> None:
        self.total_time_html.value = f"<code>{html.escape(total)}</code>"
        self.llm_time_html.value = f"<code>{html.escape(llm)}</code>"
        self.pre_time_html.value = f"<code>{html.escape(pre)}</code>"
        self.fsa_time_html.value = f"<code>{html.escape(fsa)}</code>"

    def _set_llm_runtime_labels(self, *, model_text: str, inference_text: str) -> None:
        self.llm_model_time_html.value = f"<code>{html.escape(model_text)}</code>"
        self.llm_inference_time_html.value = f"<code>{html.escape(inference_text)}</code>"

    def _render_dataframe(self, output: widgets.HTML, columns: list[str], rows: list[list[Any]], empty_message: str) -> None:
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            output.value = f"<em>{html.escape(empty_message)}</em>"
            return
        table_html = df.to_html(index=False, escape=True, border=0)
        output.value = (
            "<div style='overflow-x:auto'>"
            + table_html
            + "</div>"
        )

    def _render_step_tables(self, connect_output: widgets.HTML, process_output: widgets.HTML, lines: list[str]) -> None:
        split_rows = split_step_lines(lines)
        self._render_dataframe(connect_output, list(STEP_TABLE_COLUMNS), split_rows["connect_rows"], "No connect steps yet.")
        self._render_dataframe(process_output, list(STEP_TABLE_COLUMNS), split_rows["process_rows"], "No process steps yet.")

    def _reset_context_outputs(self) -> None:
        self.recipe_summary_html.value = "<em>No seed context has been generated yet.</em>"
        self._set_overview_status("No data yet.", tone="#6e7781")
        self._render_dataframe(self.modplant_output, ["Unit", "Inputs", "Outputs", "MaxVolume", "Resources", "Operation", "Param", "Cost"], [], "Generate a seed context to populate this table.")
        self._render_dataframe(self.schedule_output, ["Step", "Type", "Stage", "Start(s)", "Duration(s)"], [], "Generate a seed context to populate this table.")
        self._render_dataframe(self.rules_output, ["Inputs", "Reaction Type", "Reaction Param", "Result"], [], "Generate a seed context to populate this table.")

    def _reset_pipeline_outputs(self) -> None:
        self.pipeline_result = None
        self.log_lines = []
        self.llm_lines = []
        self.fsa_lines = []
        self.log_text.value = ""
        self.llm_check_text.value = ""
        self._set_check_status("Waiting for LLM check", "#6e7781")
        self._set_pipeline_status("Pipeline status", "Ready to start.", color="#6e7781")
        self._set_runtime_labels(
            total="Total: 0.00s",
            llm="LLM: -",
            pre="FSA Checker: -",
            fsa="FSA+BFS+OPT: -",
        )
        self._set_llm_runtime_labels(model_text="Model Load: -", inference_text="Inference: -")
        self.fsa_status_html.value = "<code>FSA+BFS+OPT: idle</code>"
        self._set_decision_status("Decision", "No decision is pending.", "#6e7781")
        self.decision_button_row.layout.display = "none"
        self._render_step_tables(self.llm_connect_output, self.llm_process_output, [])
        self._render_step_tables(self.fsa_connect_output, self.fsa_process_output, [])

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        disabled = not enabled
        for widget in (
            self.adapter_input,
            self.browse_button,
            self.prewarm_button,
            self.persist_files_checkbox,
            self.timeout_input,
            self.seed_input,
            self.random_seed_button,
            self.generate_button,
            self.start_button,
        ):
            widget.disabled = disabled

    def _append_log(self, message: str) -> None:
        self.log_lines.append(message)
        self.log_text.value = "\n".join(self.log_lines[-400:])

    def _start_timer_loop(self) -> None:
        self._stop_timer_loop()
        self._timer_stop_event = threading.Event()

        def _run() -> None:
            while not self._timer_stop_event.wait(0.1):
                if self.run_start > 0:
                    total = self._total_label_text()
                else:
                    total = "Total: 0.00s"

                llm_text = f"LLM: {self.llm_inference_elapsed_s:.2f}s" if self.llm_inference_elapsed_s > 0 else "LLM: 0.00s"
                if self.llm_running and self.llm_phase == "loading" and self.llm_load_start_ts > 0:
                    llm_text = "LLM: 0.00s"
                    self._set_llm_runtime_labels(
                        model_text=f"Model Load: running {time.perf_counter() - self.llm_load_start_ts:.1f}s",
                        inference_text=self.llm_inference_time_html.value.replace("<code>", "").replace("</code>", "") or "Inference: -",
                    )
                elif self.llm_running and self.llm_phase == "inference" and self.llm_start_ts > 0:
                    elapsed = time.perf_counter() - self.llm_start_ts
                    llm_text = f"LLM: {elapsed:.2f}s"
                    self._set_llm_runtime_labels(
                        model_text=self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or "Model Load: -",
                        inference_text=f"Inference: running {elapsed:.1f}s",
                    )

                pre_text = self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: -"
                if self.fsa_running and self.fsa_start_ts > 0:
                    fsa_elapsed = time.perf_counter() - self.fsa_start_ts
                    fsa_text = f"FSA+BFS+OPT: {fsa_elapsed:.2f}s"
                    self.fsa_status_html.value = f"<code>FSA+BFS+OPT: running {fsa_elapsed:.1f}s</code>"
                else:
                    fsa_text = self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: -"
                self._set_runtime_labels(total=total, llm=llm_text, pre=pre_text, fsa=fsa_text)

        self._timer_thread = threading.Thread(target=_run, daemon=True)
        self._timer_thread.start()

    def _stop_timer_loop(self) -> None:
        self._timer_stop_event.set()

    def _render_overview_payload(self, payload: dict[str, Any]) -> None:
        self.context_payload = payload
        overview = build_overview_data(payload)
        self.recipe_summary_html.value = f"<code>{html.escape(overview['recipe_summary'])}</code>"
        self._render_dataframe(
            self.modplant_output,
            overview["modplant_columns"],
            overview["modplant_rows"],
            "No ModPlant configuration rows were generated.",
        )
        self._render_dataframe(
            self.schedule_output,
            overview["schedule_columns"],
            overview["schedule_rows"],
            "No schedule rows were generated.",
        )
        self._render_dataframe(
            self.rules_output,
            overview["rule_columns"],
            overview["rule_rows"],
            "No reaction rules were generated.",
        )

    def _on_random_seed(self, _: widgets.Button) -> None:
        value = random.randint(1, 10_000_000)
        self.seed_input.value = value
        self._append_log(f"Random seed set to {value}")

    def _on_generate_context(self, _: widgets.Button) -> None:
        try:
            payload = build_context_from_seed(
                self.seed_input.value,
                persist_temp_files=bool(self.persist_files_checkbox.value),
            )
            self._render_overview_payload(payload)
            self._set_overview_status(f"Generated context for seed {self.seed_input.value}.")
            self._append_log(f"Generated context for seed={self.seed_input.value}")
        except Exception as exc:
            self._set_overview_status(str(exc), tone="#cf222e")
            self._append_log(f"Generate failed: {exc}")

    def _on_browse_adapter(self, _: widgets.Button) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(initialdir=str(self.project_root), title="Select LoRA adapter dir")
            root.destroy()
            if selected:
                self.adapter_input.value = selected
                self._set_prewarm_status("Adapter path updated from the file dialog.", tone="idle")
        except Exception as exc:
            self._set_prewarm_status(
                f"Browse dialog is unavailable here. Enter the adapter path manually. Details: {exc}",
                tone="failed",
            )

    def _run_prewarm(self, adapter_dir: str, auto: bool) -> None:
        try:
            if auto:
                self._append_log("[LLM] Auto prewarm started.")
            else:
                self._append_log("[LLM] Manual prewarm started.")
            payload = prewarm_adapter_runtime(self.project_root, adapter_dir)
            elapsed = float(payload.get("elapsed_s", 0.0))
            runtime_hit = bool(payload.get("runtime_cache_hit"))
            self._set_prewarm_status(f"Model prewarm: ready ({elapsed:.2f}s)", tone="ready")
            self._append_log(
                "[LLM] Prewarm done: "
                f"{elapsed:.2f}s | base={payload.get('base_model_name')} | device={payload.get('device')} | "
                f"cache={'hit' if runtime_hit else 'miss'}"
            )
        except Exception as exc:
            self._set_prewarm_status("Model prewarm: failed", tone="failed")
            self._append_log(f"[LLM] Prewarm failed:\n{exc}\n\n{traceback.format_exc(limit=8)}")
        finally:
            self.prewarm_button.disabled = bool(self._pipeline_thread is not None and self._pipeline_thread.is_alive())
            self._prewarm_thread = None

    def _start_prewarm(self, auto: bool) -> None:
        if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
            if not auto:
                self._append_log("Model prewarm already running.")
            return
        adapter_dir = self.adapter_input.value.strip()
        if not adapter_dir:
            self._set_prewarm_status("Enter an adapter directory first.", tone="failed")
            return
        if not Path(adapter_dir).expanduser().is_dir():
            if not auto:
                self._set_prewarm_status(f"Adapter dir not found: {adapter_dir}", tone="failed")
            return
        self.prewarm_button.disabled = True
        self._set_prewarm_status("Model prewarm: running...", tone="running")
        self._prewarm_thread = threading.Thread(target=self._run_prewarm, args=(adapter_dir, auto), daemon=True)
        self._prewarm_thread.start()

    def _on_prewarm_clicked(self, _: widgets.Button) -> None:
        self._start_prewarm(auto=False)

    def _prepare_pipeline_run(self) -> None:
        self._reset_pipeline_outputs()
        self._set_run_controls_enabled(False)
        self.run_start = time.perf_counter()
        self.total_pause_started_at = 0.0
        self.total_paused_elapsed_s = 0.0
        self.llm_running = True
        self.llm_phase = "loading"
        self.fsa_running = False
        self.llm_load_start_ts = self.run_start
        self.llm_load_elapsed_s = 0.0
        self.llm_inference_elapsed_s = 0.0
        self.llm_start_ts = 0.0
        self.fsa_start_ts = 0.0
        self._set_pipeline_status("Pipeline status", "Pipeline running in the background.", color="#1f6feb")
        self._set_runtime_labels(
            total="Total: 0.00s",
            llm="LLM: 0.00s",
            pre="FSA Checker: waiting...",
            fsa="FSA+BFS+OPT: waiting...",
        )
        self._set_llm_runtime_labels(model_text="Model Load: running 0.0s", inference_text="Inference: -")
        self.fsa_status_html.value = "<code>FSA+BFS+OPT: waiting...</code>"
        self._start_timer_loop()
        self._append_log(f"LLM settings: timeout={int(self.timeout_input.value)}s")

    def _on_start_pipeline(self, _: widgets.Button) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            self._append_log("A pipeline run is already active.")
            return
        adapter_dir = self.adapter_input.value.strip()
        if not adapter_dir:
            self._set_pipeline_status("Pipeline status", "Please enter or browse to an adapter directory first.", color="#cf222e")
            return
        self._prepare_pipeline_run()
        self._pipeline_thread = threading.Thread(target=self._run_pipeline_background, daemon=True)
        self._pipeline_thread.start()

    def _run_pipeline_background(self) -> None:
        try:
            self.pipeline_result = run_pipeline_session(
                project_root=self.project_root,
                seed=int(self.seed_input.value),
                adapter_dir=self.adapter_input.value.strip(),
                persist_temp_files=bool(self.persist_files_checkbox.value),
                llm_attempt_timeout_s=int(self.timeout_input.value),
                callbacks=PipelineEventCallbacks(
                    log=self._append_log,
                    context=self._on_context_event,
                    llm_load=self._on_llm_load_event,
                    llm_step=self._on_llm_step_event,
                    llm_preview=self._on_llm_preview_event,
                    llm_done=self._on_llm_done_event,
                    llm_check=self._on_llm_check_event,
                    fsa_started=self._on_fsa_started_event,
                    fsa_decision_needed=self._on_fsa_decision_needed_event,
                    fsa_done=self._on_fsa_done_event,
                    failed=self._on_failed_event,
                    done=self._on_done_event,
                ),
            )
        finally:
            self._pipeline_thread = None

    def _on_context_event(self, payload: dict[str, Any]) -> None:
        self._render_overview_payload(payload.get("display_context", {}))
        self._set_overview_status(f"Generated context for seed {self.seed_input.value}.")

    def _on_llm_load_event(self, payload: dict[str, Any]) -> None:
        elapsed = float(payload.get("elapsed_s", 0.0))
        cache_hit = bool(payload.get("runtime_cache_hit"))
        suffix = " (cache hit)" if cache_hit else ""
        self.llm_load_elapsed_s = elapsed
        self.llm_load_start_ts = 0.0
        self.llm_phase = "inference"
        self.llm_start_ts = time.perf_counter()
        self._set_llm_runtime_labels(model_text=f"Model Load: {elapsed:.2f}s{suffix}", inference_text="Inference: running 0.0s")
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm="LLM: 0.00s",
            pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: waiting...",
            fsa=self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: waiting...",
        )

    def _set_llm_line(self, step_idx: int, line: str) -> None:
        while len(self.llm_lines) < step_idx - 1:
            self.llm_lines.append("")
        if len(self.llm_lines) == step_idx - 1:
            self.llm_lines.append(line.strip())
        else:
            self.llm_lines[step_idx - 1] = line.strip()

    def _on_llm_step_event(self, step_idx: int, line: str, elapsed: float) -> None:
        self.llm_inference_elapsed_s = float(elapsed)
        self._set_llm_line(step_idx, line)
        self._render_step_tables(self.llm_connect_output, self.llm_process_output, self.llm_lines)
        self._set_llm_runtime_labels(
            model_text=self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or "Model Load: -",
            inference_text=f"Inference: running {elapsed:.1f}s",
        )
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm=f"LLM: {elapsed:.2f}s",
            pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: waiting...",
            fsa=self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: waiting...",
        )
        self._append_log(f"[LLM step {step_idx}] {line}")

    def _on_llm_preview_event(self, text: str, elapsed: float) -> None:
        self.llm_inference_elapsed_s = float(elapsed)
        self.llm_lines = [line.strip() for line in text.splitlines() if line.strip()]
        self._render_step_tables(self.llm_connect_output, self.llm_process_output, self.llm_lines)
        self._set_llm_runtime_labels(
            model_text=self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or "Model Load: -",
            inference_text=f"Inference: running {elapsed:.1f}s",
        )
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm=f"LLM: {elapsed:.2f}s",
            pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: waiting...",
            fsa=self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: waiting...",
        )

    def _on_llm_done_event(self, text: str, elapsed: float) -> None:
        self.llm_running = False
        self.llm_phase = "done"
        self.llm_inference_elapsed_s = float(elapsed)
        self.llm_start_ts = 0.0
        self.llm_lines = [line.strip() for line in text.splitlines() if line.strip()]
        self._render_step_tables(self.llm_connect_output, self.llm_process_output, self.llm_lines)
        self._set_llm_runtime_labels(
            model_text=self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or "Model Load: -",
            inference_text=f"Inference: {elapsed:.2f}s",
        )
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm=f"LLM: {elapsed:.2f}s",
            pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: waiting...",
            fsa=self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: waiting...",
        )

    def _on_llm_check_event(self, payload: dict[str, Any]) -> None:
        elapsed = float(payload.get("elapsed_s", 0.0))
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm=self.llm_time_html.value.replace("<code>", "").replace("</code>", "") or "LLM: -",
            pre=f"FSA Checker: {elapsed:.2f}s",
            fsa=self.fsa_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA+BFS+OPT: waiting...",
        )
        if payload.get("partial") and payload.get("auto_fsa"):
            self._set_check_status("FSA Checker Fail", "#cf222e")
            self._set_pipeline_status(
                "Pipeline status",
                "Real-time step checking failed, so FSA+BFS+OPT is starting automatically.",
                color="#cf222e",
            )
        elif payload.get("partial"):
            self._set_check_status("Checking step-by-step", "#1f6feb")
        elif payload.get("ok") is True and not (payload.get("warnings") or []):
            self._set_check_status("Pass", "#238636")
            self._set_pipeline_status(
                "Pipeline status",
                f"Flow check passed in {elapsed:.2f}s.",
                color="#238636",
            )
        elif payload.get("ok") is True:
            self._set_check_status("Pass with warnings", "#bf8700")
            self._set_pipeline_status(
                "Pipeline status",
                "Flow check passed with warnings.",
                color="#bf8700",
            )
        else:
            self._set_check_status("FSA Checker Fail", "#cf222e")
            if payload.get("auto_fsa"):
                self._set_pipeline_status(
                    "Pipeline status",
                    "Step-level FSA Checker failed, so FSA+BFS+OPT is running automatically.",
                    color="#cf222e",
                )
        self.llm_check_text.value = format_flow_check_result(payload)

    def _on_fsa_started_event(self) -> None:
        self.fsa_running = True
        self.fsa_start_ts = time.perf_counter()
        self.fsa_status_html.value = "<code>FSA+BFS+OPT: running 0.0s</code>"

    def _on_fsa_decision_needed_event(self, payload: dict[str, Any]) -> bool:
        self.llm_running = False
        self.llm_phase = "done"
        self.llm_start_ts = 0.0
        self._pause_total_timer()
        warnings = payload.get("warnings") or []
        if warnings:
            message = "The LLM solution is feasible with warnings. Decide whether to continue into FSA+BFS+OPT."
        else:
            message = "The LLM solution is feasible. Decide whether to continue into FSA+BFS+OPT."
        self._decision_value = None
        self._decision_event.clear()
        self.decision_button_row.layout.display = ""
        self._set_decision_status("Decision required", message, "#1f6feb")
        while not self._decision_event.wait(0.1):
            continue
        go_fsa = bool(self._decision_value)
        self._resume_total_timer()
        if go_fsa:
            self._set_decision_status("Decision recorded", "FSA+BFS+OPT will run.", "#238636")
        else:
            self._set_decision_status("Decision recorded", "FSA+BFS+OPT will be skipped.", "#bf8700")
        self.decision_button_row.layout.display = "none"
        return go_fsa

    def _on_fsa_done_event(self, payload: dict[str, Any]) -> None:
        self.fsa_running = False
        elapsed = float(payload.get("elapsed_s", 0.0))
        self.fsa_status_html.value = f"<code>FSA+BFS+OPT: done {elapsed:.2f}s</code>"
        self._set_runtime_labels(
            total=self.total_time_html.value.replace("<code>", "").replace("</code>", "") or "Total: 0.00s",
            llm=self.llm_time_html.value.replace("<code>", "").replace("</code>", "") or "LLM: -",
            pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: -",
            fsa=f"FSA+BFS+OPT: {elapsed:.2f}s",
        )
        self.fsa_lines = [line.strip() for line in (payload.get("reference_steps") or []) if str(line).strip()]
        self._render_step_tables(self.fsa_connect_output, self.fsa_process_output, self.fsa_lines)
        wr = payload.get("worker_result") or {}
        self._append_log(f"FSA+BFS+OPT worker status: {wr.get('status')}")
        self._append_log(f"FSA+BFS+OPT feasible: {wr.get('feasible')}")
        self._append_log(f"FSA+BFS+OPT corpus_path: {payload.get('corpus_path')}")
        if wr.get("feasible") is True:
            self._set_pipeline_status(
                "Pipeline status",
                f"Feasible FSA+BFS+OPT solution found in {elapsed:.2f}s.",
                color="#238636",
            )

    def _on_failed_event(self, message: str) -> None:
        self._stop_timer_loop()
        load_elapsed = None
        infer_elapsed = None
        if self.llm_phase == "loading" and self.llm_load_start_ts > 0:
            load_elapsed = time.perf_counter() - self.llm_load_start_ts
        elif self.llm_phase == "inference" and self.llm_start_ts > 0:
            infer_elapsed = time.perf_counter() - self.llm_start_ts
        self.llm_running = False
        self.fsa_running = False
        self.llm_load_start_ts = 0.0
        self.llm_start_ts = 0.0
        if self.llm_phase == "loading":
            self._set_llm_runtime_labels(
                model_text=(f"Model Load: failed after {load_elapsed:.2f}s" if load_elapsed is not None else "Model Load: failed"),
                inference_text=self.llm_inference_time_html.value.replace("<code>", "").replace("</code>", "") or "Inference: -",
            )
        elif self.llm_phase == "inference":
            self._set_llm_runtime_labels(
                model_text=self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or "Model Load: -",
                inference_text=(f"Inference: failed after {infer_elapsed:.2f}s" if infer_elapsed is not None else "Inference: failed"),
            )
        self.llm_phase = "failed"
        self._set_check_status("Fail", "#cf222e")
        self.fsa_status_html.value = "<code>FSA+BFS+OPT: idle</code>"
        self._set_pipeline_status("Pipeline status", "Pipeline failed. See the runtime log for details.", color="#cf222e")
        self._append_log(message)
        self._set_run_controls_enabled(True)

    def _on_done_event(self, payload: dict[str, Any]) -> None:
        self._stop_timer_loop()
        self.llm_running = False
        self.fsa_running = False
        self.llm_phase = "done"
        self.llm_load_start_ts = 0.0
        self.llm_start_ts = 0.0
        runtime = payload.get("runtime") or {}
        self.llm_load_elapsed_s = float(runtime.get("model_load_elapsed_s", self.llm_load_elapsed_s))
        self.llm_inference_elapsed_s = float(payload.get("llm_elapsed_s", self.llm_inference_elapsed_s))
        self._set_run_controls_enabled(True)
        if self.run_start > 0:
            self._set_runtime_labels(
                total=self._total_label_text(),
                llm=f"LLM: {self.llm_inference_elapsed_s:.2f}s",
                pre=self.pre_time_html.value.replace("<code>", "").replace("</code>", "") or "FSA Checker: -",
                fsa=(
                    "FSA+BFS+OPT: skipped"
                    if payload.get("fsa_skipped")
                    else f"FSA+BFS+OPT: {float(payload.get('fsa_elapsed_s', 0.0)):.2f}s"
                ),
            )
        self._set_llm_runtime_labels(
            model_text=(self.llm_model_time_html.value.replace("<code>", "").replace("</code>", "") or f"Model Load: {self.llm_load_elapsed_s:.2f}s"),
            inference_text=f"Inference: {self.llm_inference_elapsed_s:.2f}s",
        )
        if payload.get("fsa_skipped"):
            self.fsa_status_html.value = "<code>FSA+BFS+OPT: skipped by decision</code>"
            self._append_log("FSA+BFS+OPT was skipped by user decision.")
            self._set_pipeline_status(
                "Pipeline status",
                "Pipeline finished after the feasible LLM result and a user decision to skip FSA+BFS+OPT.",
                color="#238636",
            )
        else:
            self.fsa_status_html.value = f"<code>FSA+BFS+OPT: done {float(payload.get('fsa_elapsed_s', 0.0)):.2f}s</code>"
            self._set_pipeline_status("Pipeline status", "Pipeline finished successfully.", color="#238636")
        self._append_log(f"Done: {payload}")

    def _on_decision_yes(self, _: widgets.Button) -> None:
        self._decision_value = True
        self._decision_event.set()

    def _on_decision_no(self, _: widgets.Button) -> None:
        self._decision_value = False
        self._decision_event.set()


__all__ = ["ModPlantNotebookUI"]
