from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEvent, QEasingCurve, QObject, QPropertyAnimation, QThread, QTimer, Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QWidget
from qfluentwidgets import (
    FluentIcon,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    NavigationItemPosition,
    Theme,
    setTheme,
    setThemeColor,
)

from .pages import FSAPage, HomePage, LLMPage, LogPage, SettingsPage
from .pipeline import DEFAULT_ADAPTER_REL_PATH, DEFAULT_PERSIST_TEMP_FILES, build_context_from_seed
from .session import build_overview_data, format_flow_check_result, split_step_lines
from .workers import ModelPrewarmWorker, PipelineWorker

class WabenPlannerWindow(FluentWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.persist_temp_files = DEFAULT_PERSIST_TEMP_FILES
        self.worker: PipelineWorker | None = None
        self.prewarm_worker: ModelPrewarmWorker | None = None
        self._prewarm_auto_mode = False
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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._nav_click_targets: dict[QObject, QWidget] = {}
        self._build_window()

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

    def _stop_thread(self, thread_obj: QThread | None, name: str, timeout_ms: int = 300) -> None:
        if thread_obj is None:
            return
        if not thread_obj.isRunning():
            return
        if hasattr(thread_obj, "request_stop"):
            try:
                thread_obj.request_stop()  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            thread_obj.requestInterruption()
        thread_obj.quit()
        if thread_obj.wait(timeout_ms):
            self._append_log(f"[Shutdown] {name} stopped.")
            return
        self._append_log(f"[Shutdown] {name} did not stop in time, forcing terminate().")
        thread_obj.terminate()
        thread_obj.wait(300)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        self._stop_thread(self.worker, "PipelineWorker")
        self._stop_thread(self.prewarm_worker, "ModelPrewarmWorker")
        super().closeEvent(event)

    def _disable_switch_animation(self) -> None:
        stacked = getattr(self, "stackedWidget", None)
        if stacked is None:
            return

        disabled_any = False
        for obj in (stacked, getattr(stacked, "view", None)):
            if obj is None:
                continue
            method = getattr(obj, "setAnimationEnabled", None)
            if callable(method):
                try:
                    method(False)
                    disabled_any = True
                except Exception:
                    pass

        targets: list[Any] = [stacked, getattr(stacked, "view", None)]
        for name in ("ani", "animation", "switchAni", "slideAni", "popAni", "moveAni"):
            obj = getattr(stacked, name, None)
            if obj is not None:
                targets.append(obj)
        view = getattr(stacked, "view", None)
        if view is not None:
            for name in ("ani", "animation", "switchAni", "slideAni", "popAni", "moveAni", "_ani"):
                obj = getattr(view, name, None)
                if obj is not None:
                    targets.append(obj)
        try:
            targets.extend(stacked.findChildren(QObject))
        except Exception:
            pass
        if view is not None:
            try:
                targets.extend(view.findChildren(QObject))
            except Exception:
                pass

        seen_ids: set[int] = set()
        for obj in targets:
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)

            for method_name in ("setDuration", "setAnimationDuration", "setAniDuration"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method(0)
                        disabled_any = True
                    except Exception:
                        pass
            set_curve = getattr(obj, "setEasingCurve", None)
            if callable(set_curve):
                try:
                    set_curve(QEasingCurve.Type.Linear)
                    disabled_any = True
                except Exception:
                    pass
            if isinstance(obj, QPropertyAnimation):
                try:
                    obj.setDuration(0)
                    obj.setEasingCurve(QEasingCurve.Type.Linear)
                    disabled_any = True
                except Exception:
                    pass

        for obj in self.findChildren(QPropertyAnimation):
            try:
                obj.setDuration(0)
                obj.setEasingCurve(QEasingCurve.Type.Linear)
                disabled_any = True
            except Exception:
                pass

        if disabled_any:
            self._append_log("[UI] Page switch animation disabled.")

    def _build_window(self) -> None:
        self.setWindowTitle("ModPlant-LLM")
        self.resize(1600, 980)
        setTheme(Theme.DARK)
        setThemeColor("#00A3A3")
        self.home_page = HomePage(self)
        self.llm_page = LLMPage(self)
        self.fsa_page = FSAPage(self)
        self.log_page = LogPage(self)
        self.settings_page = SettingsPage(self)
        self.settings_page.adapter_edit.setText(str((self.project_root / DEFAULT_ADAPTER_REL_PATH).resolve()))
        self.settings_page.persist_files_switch.setChecked(self.persist_temp_files)

        self.home_nav_item = self.addSubInterface(self.home_page, FluentIcon.HOME, "Home")
        self.llm_nav_item = self.addSubInterface(self.llm_page, FluentIcon.ROBOT, "LLM")
        self.fsa_nav_item = self.addSubInterface(self.fsa_page, FluentIcon.IOT, "FSA+BFS+OPT")
        self.log_nav_item = self.addSubInterface(self.log_page, FluentIcon.DOCUMENT, "Log")
        self.settings_nav_item = self.addSubInterface(self.settings_page, FluentIcon.SETTING, "Settings", NavigationItemPosition.BOTTOM)

        self._install_nav_click_forwarder(self.home_nav_item, self.home_page)
        self._install_nav_click_forwarder(self.llm_nav_item, self.llm_page)
        self._install_nav_click_forwarder(self.fsa_nav_item, self.fsa_page)
        self._install_nav_click_forwarder(self.log_nav_item, self.log_page)
        self._install_nav_click_forwarder(self.settings_nav_item, self.settings_page)

        self.settings_page.btn_adapter.clicked.connect(self._choose_adapter_dir)
        self.home_page.btn_random_seed.clicked.connect(self._randomize_seed)
        self.home_page.btn_generate.clicked.connect(self._generate_seed_context)
        self.home_page.start_btn.clicked.connect(self._start_pipeline)
        self.settings_page.persist_files_switch.checkedChanged.connect(self._on_persist_files_changed)
        self._disable_switch_animation()
        QTimer.singleShot(0, self._disable_switch_animation)
        QTimer.singleShot(250, self._disable_switch_animation)
        self.switchTo(self.home_page)
        QTimer.singleShot(150, self._auto_prewarm_default_adapter)

    def _install_nav_click_forwarder(self, nav_item: QObject, page: QWidget) -> None:
        for click_target in (nav_item, getattr(nav_item, "itemWidget", None)):
            if click_target is None:
                continue
            self._nav_click_targets[click_target] = page
            click_target.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        target_page = getattr(self, "_nav_click_targets", {}).get(obj)
        if target_page is not None and event.type() == QEvent.Type.MouseButtonPress:
            button = getattr(event, "button", None)
            if callable(button):
                try:
                    if button() == Qt.MouseButton.LeftButton:
                        self.switchTo(target_page)
                        self.navigationInterface.setCurrentItem(target_page.objectName())
                except Exception:
                    pass
        return super().eventFilter(obj, event)

    def _on_persist_files_changed(self, checked: bool) -> None:
        self.persist_temp_files = bool(checked)
        self._append_log(f"Persist temporary artifacts set to {self.persist_temp_files}")

    def _append_log(self, msg: str) -> None:
        self.log_page.log_text.appendPlainText(msg)
        self.log_page.log_text.verticalScrollBar().setValue(self.log_page.log_text.verticalScrollBar().maximum())

    def _tick(self) -> None:
        if self.run_start > 0:
            self.home_page.total_time_label.setText(self._total_label_text())
        if self.llm_running:
            if self.llm_phase == "loading" and self.llm_load_start_ts > 0:
                load_elapsed = time.perf_counter() - self.llm_load_start_ts
                self.home_page.llm_time_label.setText("LLM: 0.00s")
                self.llm_page.set_model_loading(time.perf_counter() - self.llm_load_start_ts)
            elif self.llm_phase == "inference" and self.llm_start_ts > 0:
                infer_elapsed = time.perf_counter() - self.llm_start_ts
                self.home_page.llm_time_label.setText(
                    f"LLM: {infer_elapsed:.2f}s"
                )
                self.llm_page.set_inference_running(infer_elapsed)
        if self.fsa_running and self.fsa_start_ts > 0:
            self.fsa_page.set_busy(True, time.perf_counter() - self.fsa_start_ts)

    def _choose_adapter_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select LoRA adapter dir", str(self.project_root))
        if d:
            self.settings_page.adapter_edit.setText(d)
            self._start_model_prewarm(d, auto=False)

    def _auto_prewarm_default_adapter(self) -> None:
        adapter_dir = self.settings_page.adapter_edit.text().strip()
        if adapter_dir:
            self._start_model_prewarm(adapter_dir, auto=True)

    def _start_model_prewarm(self, adapter_dir: str, auto: bool) -> None:
        if self.prewarm_worker is not None and self.prewarm_worker.isRunning():
            if not auto:
                self._append_log("Model prewarm already running.")
            return
        if not Path(adapter_dir).expanduser().is_dir():
            if not auto:
                QMessageBox.warning(self, "Invalid Adapter", f"Adapter dir not found:\n{adapter_dir}")
            return
        self._prewarm_auto_mode = auto
        self.settings_page.prewarm_status.setText("Model prewarm: running...")
        if auto:
            self._append_log("[LLM] Auto prewarm started.")
        else:
            self._append_log("[LLM] Manual prewarm started.")
        self.prewarm_worker = ModelPrewarmWorker(self.project_root, adapter_dir)
        self.prewarm_worker.done_signal.connect(self._on_prewarm_done)
        self.prewarm_worker.failed_signal.connect(self._on_prewarm_failed)
        self.prewarm_worker.start()

    def _on_prewarm_done(self, payload: dict[str, Any]) -> None:
        elapsed = float(payload.get("elapsed_s", 0.0))
        runtime_hit = bool(payload.get("runtime_cache_hit"))
        self.settings_page.prewarm_status.setText(f"Model prewarm: ready ({elapsed:.2f}s)")
        self._append_log(
            "[LLM] Prewarm done: "
            f"{elapsed:.2f}s | base={payload.get('base_model_name')} | device={payload.get('device')} | "
            f"cache={'hit' if runtime_hit else 'miss'}"
        )
        self.prewarm_worker = None
        self._prewarm_auto_mode = False

    def _on_prewarm_failed(self, msg: str) -> None:
        self.settings_page.prewarm_status.setText("Model prewarm: failed")
        self._append_log(f"[LLM] Prewarm failed:\n{msg}")
        if not self._prewarm_auto_mode:
            QMessageBox.critical(self, "Model Prewarm Failed", msg)
        self.prewarm_worker = None
        self._prewarm_auto_mode = False

    def _randomize_seed(self) -> None:
        value = random.randint(1, 10_000_000)
        self.home_page.seed_spin.setValue(value)
        self._append_log(f"Random seed set to {value}")

    def _set_table_rows(self, table: QTableWidget, rows: list[list[Any]]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                table.setItem(r, c, item)

    def _set_split_step_tables_from_lines(
        self,
        connect_table: QTableWidget,
        process_table: QTableWidget,
        lines: list[str],
    ) -> None:
        split_rows = split_step_lines(lines)
        self._set_table_rows(connect_table, split_rows["connect_rows"])
        self._set_table_rows(process_table, split_rows["process_rows"])
        for table in (connect_table, process_table):
            for r in range(table.rowCount()):
                item = table.item(r, 0)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def _append_step_row_split(
        self,
        connect_table: QTableWidget,
        process_table: QTableWidget,
        line: str,
    ) -> None:
        split_rows = split_step_lines([line])
        if split_rows["connect_rows"]:
            target = connect_table
            row = split_rows["connect_rows"][0]
        elif split_rows["process_rows"]:
            target = process_table
            row = split_rows["process_rows"][0]
        else:
            return
        row[0] = str(target.rowCount() + 1)
        r = target.rowCount()
        target.insertRow(r)
        for c, val in enumerate(row):
            item = QTableWidgetItem(str(val))
            if c == 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            target.setItem(r, c, item)

    def _populate_overview(self, payload: dict[str, Any]) -> None:
        overview = build_overview_data(payload)
        self._set_table_rows(self.home_page.ModPlant_table, overview["modplant_rows"])
        self._set_table_rows(self.home_page.schedule_table, overview["schedule_rows"])
        self._set_table_rows(self.home_page.rules_table, overview["rule_rows"])
        self.home_page.recipe_summary.setText(overview["recipe_summary"])

    def _generate_seed_context(self) -> None:
        try:
            payload = build_context_from_seed(
                self.home_page.seed_spin.value(),
                persist_temp_files=self.persist_temp_files,
            )
            self._populate_overview(payload)
            self._append_log(f"Generated context for seed={self.home_page.seed_spin.value()}")
        except Exception as e:
            QMessageBox.critical(self, "Generate failed", str(e))

    def _start_pipeline(self) -> None:
        adapter_dir = self.settings_page.adapter_edit.text().strip()
        if not adapter_dir:
            QMessageBox.warning(self, "Missing Adapter", "Please select adapter directory first.")
            return

        self.log_page.log_text.clear()
        self.llm_page.llm_connect_table.setRowCount(0)
        self.llm_page.llm_process_table.setRowCount(0)
        self.llm_page.check_text.clear()
        self.llm_page.check_status_dot.setStyleSheet("background:#808080;border-radius:7px;")
        self.llm_page.check_status_text.setText("Waiting for LLM check...")
        self.fsa_page.fsa_connect_table.setRowCount(0)
        self.fsa_page.fsa_process_table.setRowCount(0)
        self.home_page.llm_time_label.setText("LLM: 0.00s")
        self.home_page.pre_time_label.setText("FSA Checker: waiting...")
        self.home_page.fsa_time_label.setText("FSA+BFS+OPT: waiting...")

        self.home_page.start_btn.setEnabled(False)
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
        self.llm_page.reset_timing()
        self.llm_page.set_model_loading(0.0)
        self.fsa_page.set_busy(False, None)
        self.timer.start(100)
        llm_attempt_timeout_s = int(self.settings_page.llm_attempt_timeout_spin.value())
        self._append_log(
            f"LLM settings: timeout={llm_attempt_timeout_s}s"
        )

        self.worker = PipelineWorker(
            project_root=self.project_root,
            seed=self.home_page.seed_spin.value(),
            adapter_dir=adapter_dir,
            persist_temp_files=self.persist_temp_files,
            llm_attempt_timeout_s=llm_attempt_timeout_s,
        )
        self.worker.log_signal.connect(self._append_log)
        self.worker.context_signal.connect(self._on_context)
        self.worker.llm_load_signal.connect(self._on_llm_load_done)
        self.worker.llm_step_signal.connect(self._on_llm_step)
        self.worker.llm_preview_signal.connect(self._on_llm_preview)
        self.worker.llm_done_signal.connect(self._on_llm_done)
        self.worker.llm_check_signal.connect(self._on_llm_check)
        self.worker.fsa_started_signal.connect(self._on_fsa_started)
        self.worker.fsa_decision_needed_signal.connect(self._on_fsa_decision_needed)
        self.worker.fsa_signal.connect(self._on_fsa_done)
        self.worker.failed_signal.connect(self._on_failed)
        self.worker.done_signal.connect(self._on_done)
        self.worker.start()

    def _on_llm_load_done(self, payload: dict[str, Any]) -> None:
        elapsed = float(payload.get("elapsed_s", 0.0))
        cache_hit = bool(payload.get("runtime_cache_hit"))
        self.llm_load_elapsed_s = elapsed
        self.llm_load_start_ts = 0.0
        self.llm_phase = "inference"
        self.llm_start_ts = time.perf_counter()
        self.home_page.llm_time_label.setText("LLM: 0.00s")
        self.llm_page.set_model_loaded(elapsed, cache_hit=cache_hit)
        self.llm_page.set_inference_running(0.0)

    def _on_context(self, payload: dict[str, Any]) -> None:
        self._populate_overview(payload.get("display_context", {}))

    def _on_llm_step(self, step_idx: int, line: str, elapsed: float) -> None:
        self.llm_inference_elapsed_s = float(elapsed)
        self.home_page.llm_time_label.setText(f"LLM: {elapsed:.2f}s")
        self.llm_page.set_inference_running(elapsed)
        self._append_log(f"[LLM step {step_idx}] {line}")
        self._append_step_row_split(
            self.llm_page.llm_connect_table,
            self.llm_page.llm_process_table,
            line,
        )

    def _on_llm_done(self, text: str, elapsed: float) -> None:
        self.llm_running = False
        self.llm_phase = "done"
        self.llm_inference_elapsed_s = float(elapsed)
        self.llm_start_ts = 0.0
        self.home_page.llm_time_label.setText(f"LLM: {elapsed:.2f}s")
        self.llm_page.set_inference_done(elapsed)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self._set_split_step_tables_from_lines(
            self.llm_page.llm_connect_table,
            self.llm_page.llm_process_table,
            lines,
        )

    def _on_llm_preview(self, text: str, elapsed: float) -> None:
        self.llm_inference_elapsed_s = float(elapsed)
        self.home_page.llm_time_label.setText(f"LLM: {elapsed:.2f}s")
        self.llm_page.set_inference_running(elapsed)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self._set_split_step_tables_from_lines(
            self.llm_page.llm_connect_table,
            self.llm_page.llm_process_table,
            lines,
        )

    def _on_fsa_started(self) -> None:
        self.fsa_running = True
        self.fsa_start_ts = time.perf_counter()
        self.fsa_page.set_busy(True, 0.0)

    def _on_llm_check(self, payload: dict[str, Any]) -> None:
        self.home_page.pre_time_label.setText(f"FSA Checker: {float(payload.get('elapsed_s', 0.0)):.2f}s")
        if payload.get("partial") and payload.get("auto_fsa"):
            self.llm_page.check_status_dot.setStyleSheet("background:#f85149;border-radius:7px;")
            self.llm_page.check_status_text.setText("FSA Checker Fail")
            self.llm_page.check_text.setPlainText(format_flow_check_result(payload))
            InfoBar.error(
                title="FSA Checker Failed",
                content="Real-time FSA Checker step check failed. LLM stopped and FSA+BFS+OPT is running automatically.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3800,
                parent=self,
            )
            return
        if payload.get("partial"):
            self.llm_page.check_status_dot.setStyleSheet("background:#58a6ff;border-radius:7px;")
            self.llm_page.check_status_text.setText("Checking step-by-step")
            self.llm_page.check_text.setPlainText(format_flow_check_result(payload))
            return
        if payload.get("ok") is True and not (payload.get("warnings") or []):
            self.llm_page.check_status_dot.setStyleSheet("background:#2ea043;border-radius:7px;")
            self.llm_page.check_status_text.setText("Pass")
            InfoBar.success(
                title="LLM Check Passed",
                content=f"Flow check passed in {float(payload.get('elapsed_s', 0.0)):.2f}s.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2600,
                parent=self,
            )
        elif payload.get("ok") is True:
            self.llm_page.check_status_dot.setStyleSheet("background:#d29922;border-radius:7px;")
            self.llm_page.check_status_text.setText("Pass with warnings")
            InfoBar.success(
                title="LLM Check Passed",
                content="Flow check passed with warnings.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2800,
                parent=self,
            )
        else:
            self.llm_page.check_status_dot.setStyleSheet("background:#f85149;border-radius:7px;")
            self.llm_page.check_status_text.setText("FSA Checker Fail")
            if payload.get("auto_fsa"):
                InfoBar.error(
                    title="FSA Checker Failed",
                    content="Step-level FSA Checker failed. LLM stopped and FSA+BFS+OPT is running automatically.",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=3600,
                    parent=self,
                )
        self.llm_page.check_text.setPlainText(format_flow_check_result(payload))

    def _on_fsa_decision_needed(self, payload: dict[str, Any]) -> None:
        if self.worker is None:
            return
        warnings = payload.get("warnings") or []
        if warnings:
            title = "LLM Solution Is Feasible (With Warnings)"
            content = "The LLM solution is feasible (with warnings). Do you still want to run FSA+BFS+OPT?"
        else:
            title = "LLM Solution Is Feasible"
            content = "The LLM solution is feasible. Do you still want to run FSA+BFS+OPT?"
        self._pause_total_timer()
        reply = QMessageBox.question(
            self,
            title,
            content,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self._resume_total_timer()
        go_fsa = reply == QMessageBox.StandardButton.Yes
        self.worker.set_fsa_decision(go_fsa)
        if go_fsa:
            self.switchTo(self.fsa_page)

    def _on_fsa_done(self, payload: dict[str, Any]) -> None:
        self.fsa_running = False
        self.home_page.fsa_time_label.setText(f"FSA+BFS+OPT: {float(payload.get('elapsed_s', 0.0)):.2f}s")
        self.fsa_page.set_busy(False, float(payload.get("elapsed_s", 0.0)))

        ref_steps = payload.get("reference_steps") or []
        self._set_split_step_tables_from_lines(
            self.fsa_page.fsa_connect_table,
            self.fsa_page.fsa_process_table,
            ref_steps,
        )

        wr = payload.get("worker_result") or {}
        if wr.get("feasible") is True:
            InfoBar.success(
                title="FSA+BFS+OPT Succeeded",
                content=f"Feasible solution found in {float(payload.get('elapsed_s', 0.0)):.2f}s.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2800,
                parent=self,
            )
        self._append_log(f"FSA+BFS+OPT worker status: {wr.get('status')}")
        self._append_log(f"FSA+BFS+OPT feasible: {wr.get('feasible')}")
        self._append_log(f"FSA+BFS+OPT corpus_path: {payload.get('corpus_path')}")

    def _on_failed(self, msg: str) -> None:
        self.timer.stop()
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
            self.llm_page.set_model_load_failed(load_elapsed)
        elif self.llm_phase == "inference":
            self.llm_page.set_inference_failed(infer_elapsed)
        self.llm_phase = "failed"
        self.fsa_page.set_busy(False, None)
        self.home_page.start_btn.setEnabled(True)
        self.llm_page.check_status_dot.setStyleSheet("background:#f85149;border-radius:7px;")
        self.llm_page.check_status_text.setText("Fail")
        self._append_log(msg)
        QMessageBox.critical(self, "Pipeline Failed", msg)

    def _on_done(self, payload: dict[str, Any]) -> None:
        self.timer.stop()
        self.llm_running = False
        self.fsa_running = False
        self.llm_phase = "done"
        self.llm_load_start_ts = 0.0
        self.llm_start_ts = 0.0
        runtime = payload.get("runtime") or {}
        self.llm_load_elapsed_s = float(runtime.get("model_load_elapsed_s", self.llm_load_elapsed_s))
        self.llm_inference_elapsed_s = float(payload.get("llm_elapsed_s", self.llm_inference_elapsed_s))
        self.home_page.start_btn.setEnabled(True)
        self.home_page.total_time_label.setText(self._total_label_text())
        self.home_page.llm_time_label.setText(
            f"LLM: {self.llm_inference_elapsed_s:.2f}s"
        )
        self.llm_page.set_inference_done(self.llm_inference_elapsed_s)
        if payload.get("fsa_skipped"):
            self.fsa_page.set_busy(False, None)
        else:
            self.fsa_page.set_busy(False, float(payload.get("fsa_elapsed_s", 0.0)))
        if payload.get("fsa_skipped"):
            self.home_page.fsa_time_label.setText("FSA+BFS+OPT: skipped")
            self._append_log("FSA+BFS+OPT was skipped by user decision.")
        self._append_log(f"Done: {payload}")

__all__ = ["WabenPlannerWindow"]
