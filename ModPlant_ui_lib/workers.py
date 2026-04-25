from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from .pipeline import DEFAULT_LLM_ATTEMPT_TIMEOUT_S, DEFAULT_PERSIST_TEMP_FILES, DEFAULT_UI_SEED
from .session import PipelineEventCallbacks, prewarm_adapter_runtime, run_pipeline_session


class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    context_signal = pyqtSignal(dict)
    llm_load_signal = pyqtSignal(dict)
    llm_step_signal = pyqtSignal(int, str, float)
    llm_preview_signal = pyqtSignal(str, float)
    llm_done_signal = pyqtSignal(str, float)
    llm_check_signal = pyqtSignal(dict)
    fsa_started_signal = pyqtSignal()
    fsa_decision_needed_signal = pyqtSignal(dict)
    fsa_signal = pyqtSignal(dict)
    failed_signal = pyqtSignal(str)
    done_signal = pyqtSignal(dict)

    def __init__(
        self,
        project_root: Path,
        seed: int,
        adapter_dir: str,
        persist_temp_files: bool = DEFAULT_PERSIST_TEMP_FILES,
        llm_attempt_timeout_s: int = DEFAULT_LLM_ATTEMPT_TIMEOUT_S,
    ):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.seed = int(seed)
        self.adapter_dir = Path(adapter_dir).expanduser().resolve()
        self.persist_temp_files = bool(persist_temp_files)
        self.llm_attempt_timeout_s = max(1, int(llm_attempt_timeout_s))
        self._decision_event = threading.Event()
        self._run_fsa = True

    def _emit_log(self, msg: str) -> None:
        self.log_signal.emit(msg)

    def _await_fsa_decision(self, payload: dict[str, Any]) -> bool:
        self._decision_event.clear()
        self.fsa_decision_needed_signal.emit(payload)
        while not self._decision_event.wait(0.1):
            if self.isInterruptionRequested():
                return False
        return bool(self._run_fsa)

    def set_fsa_decision(self, run_fsa: bool) -> None:
        self._run_fsa = bool(run_fsa)
        self._decision_event.set()

    def request_stop(self) -> None:
        self.requestInterruption()
        self._decision_event.set()

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            run_pipeline_session(
                project_root=self.project_root,
                seed=self.seed,
                adapter_dir=self.adapter_dir,
                persist_temp_files=self.persist_temp_files,
                llm_attempt_timeout_s=self.llm_attempt_timeout_s,
                callbacks=PipelineEventCallbacks(
                    log=self.log_signal.emit,
                    context=self.context_signal.emit,
                    llm_load=self.llm_load_signal.emit,
                    llm_step=self.llm_step_signal.emit,
                    llm_preview=self.llm_preview_signal.emit,
                    llm_done=self.llm_done_signal.emit,
                    llm_check=self.llm_check_signal.emit,
                    fsa_started=self.fsa_started_signal.emit,
                    fsa_decision_needed=self._await_fsa_decision,
                    fsa_done=self.fsa_signal.emit,
                    failed=self.failed_signal.emit,
                    done=self.done_signal.emit,
                ),
                stop_cb=self.isInterruptionRequested,
            )
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            self.failed_signal.emit(f"{exc}\n\n{tb}")


class ModelPrewarmWorker(QThread):
    done_signal = pyqtSignal(dict)
    failed_signal = pyqtSignal(str)

    def __init__(self, project_root: Path, adapter_dir: str):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.adapter_dir = Path(adapter_dir).expanduser().resolve()

    def request_stop(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            if not self.adapter_dir.is_dir():
                raise FileNotFoundError(f"Adapter dir not found: {self.adapter_dir}")
            runtime_info = prewarm_adapter_runtime(
                project_root=self.project_root,
                adapter_dir=self.adapter_dir,
                seed=DEFAULT_UI_SEED,
            )
            if self.isInterruptionRequested():
                return
            self.done_signal.emit(runtime_info)
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            self.failed_signal.emit(f"{exc}\n\n{tb}")


__all__ = ["ModelPrewarmWorker", "PipelineWorker"]
