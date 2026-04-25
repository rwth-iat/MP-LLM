from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CaptionLabel,
    PrimaryPushButton,
    PushButton,
    SwitchButton,
    TitleLabel,
)

from .pipeline import DEFAULT_LLM_ATTEMPT_TIMEOUT_S, DEFAULT_UI_SEED
from .widgets import BusySpinner

class HomePage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("home_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        root.addWidget(TitleLabel("ModPlant-LLM", self))
        root.addWidget(CaptionLabel("Seed -> LLM inference -> FSA Checker -> FSA+BFS+OPT", self))

        seed_card = CardWidget(self)
        seed_layout = QGridLayout(seed_card)
        seed_layout.setHorizontalSpacing(10)
        seed_layout.setVerticalSpacing(8)

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(1, 10_000_000)
        self.seed_spin.setValue(DEFAULT_UI_SEED)
        self.btn_random_seed = PushButton("Random Seed", self)
        self.btn_random_seed.setFixedWidth(120)
        self.btn_generate = PushButton("Generate From Seed", self)

        seed_layout.addWidget(BodyLabel("Seed", self), 0, 0)
        seed_layout.addWidget(self.seed_spin, 0, 1)
        seed_layout.addWidget(self.btn_random_seed, 0, 2)
        seed_layout.addWidget(self.btn_generate, 0, 3)
        seed_layout.setColumnStretch(4, 1)
        root.addWidget(seed_card)

        center = QHBoxLayout()
        center.setSpacing(10)

        ModPlant_card = CardWidget(self)
        ModPlant_layout = QVBoxLayout(ModPlant_card)
        ModPlant_layout.addWidget(BodyLabel("ModPlant Configuration", self))
        self.ModPlant_table = QTableWidget(self)
        self.ModPlant_table.setColumnCount(8)
        self.ModPlant_table.setHorizontalHeaderLabels(
            ["Unit", "Inputs", "Outputs", "MaxVolume", "Resources", "Operation", "Param", "Cost"]
        )
        self.ModPlant_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ModPlant_table.setAlternatingRowColors(True)
        ModPlant_layout.addWidget(self.ModPlant_table, 1)

        recipe_card = CardWidget(self)
        recipe_layout = QVBoxLayout(recipe_card)
        recipe_layout.addWidget(BodyLabel("Recipe Details", self))
        self.recipe_summary = CaptionLabel("No data yet", self)
        recipe_layout.addWidget(self.recipe_summary)
        recipe_layout.addWidget(BodyLabel("Schedule", self))
        self.schedule_table = QTableWidget(self)
        self.schedule_table.setColumnCount(5)
        self.schedule_table.setHorizontalHeaderLabels(["Step", "Type", "Stage", "Start(s)", "Duration(s)"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_table.setAlternatingRowColors(True)
        recipe_layout.addWidget(self.schedule_table, 1)
        recipe_layout.addWidget(BodyLabel("Reaction Rules", self))
        self.rules_table = QTableWidget(self)
        self.rules_table.setColumnCount(4)
        self.rules_table.setHorizontalHeaderLabels(["Inputs", "Reaction Type", "Reaction Param", "Result"])
        self.rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rules_table.setAlternatingRowColors(True)
        recipe_layout.addWidget(self.rules_table, 1)

        center.addWidget(ModPlant_card, 1)
        center.addWidget(recipe_card, 1)
        root.addLayout(center, 1)

        action_row = QHBoxLayout()
        self.start_btn = PrimaryPushButton("Start Calculation", self)
        action_row.addWidget(self.start_btn)
        self.total_time_label = QLabel("Total: 0.00s", self)
        self.llm_time_label = QLabel("LLM: -", self)
        self.pre_time_label = QLabel("FSA Checker: -", self)
        self.fsa_time_label = QLabel("FSA+BFS+OPT: -", self)
        self.status_labels = (
            self.total_time_label,
            self.llm_time_label,
            self.pre_time_label,
            self.fsa_time_label,
        )
        for label in self.status_labels:
            label.setStyleSheet("color: #ffffff;")
        action_row.addWidget(self.total_time_label)
        action_row.addWidget(self.llm_time_label)
        action_row.addWidget(self.pre_time_label)
        action_row.addWidget(self.fsa_time_label)
        action_row.addStretch(1)
        root.addLayout(action_row)

class LLMPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("llm_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        title_row = QHBoxLayout()
        title_row.addWidget(TitleLabel("LLM", self))
        title_row.addStretch(1)
        self.llm_busy_spinner = BusySpinner(self, size=16, color="#58a6ff")
        self.llm_model_time = CaptionLabel("Model Load: -", self)
        self.llm_inference_time = CaptionLabel("Inference: -", self)
        timing_col = QVBoxLayout()
        timing_col.setSpacing(0)
        timing_col.addWidget(self.llm_model_time)
        timing_col.addWidget(self.llm_inference_time)
        title_row.addWidget(self.llm_busy_spinner)
        title_row.addLayout(timing_col)
        root.addLayout(title_row)
        self._llm_model_busy = False
        self._llm_inference_busy = False
        root.addWidget(BodyLabel("LLM Output - Connect", self))
        self.llm_connect_table = QTableWidget(self)
        self.llm_connect_table.setColumnCount(4)
        self.llm_connect_table.setHorizontalHeaderLabels(["Step", "Operation", "Cost", "Duration"])
        llm_connect_header = self.llm_connect_table.horizontalHeader()
        llm_connect_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        llm_connect_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        llm_connect_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        llm_connect_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.llm_connect_table.setAlternatingRowColors(True)
        root.addWidget(self.llm_connect_table, 1)
        root.addWidget(BodyLabel("LLM Output - Process", self))
        self.llm_process_table = QTableWidget(self)
        self.llm_process_table.setColumnCount(4)
        self.llm_process_table.setHorizontalHeaderLabels(["Step", "Operation", "Cost", "Duration"])
        llm_process_header = self.llm_process_table.horizontalHeader()
        llm_process_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        llm_process_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        llm_process_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        llm_process_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.llm_process_table.setAlternatingRowColors(True)
        root.addWidget(self.llm_process_table, 2)
        root.addWidget(BodyLabel("Check Result", self))
        status_row = QHBoxLayout()
        self.check_status_dot = QLabel(self)
        self.check_status_dot.setFixedSize(14, 14)
        self.check_status_dot.setStyleSheet("background:#808080;border-radius:7px;")
        self.check_status_text = BodyLabel("Not checked", self)
        status_row.addWidget(self.check_status_dot)
        status_row.addWidget(self.check_status_text)
        status_row.addStretch(1)
        root.addLayout(status_row)
        self.check_text = QPlainTextEdit(self)
        self.check_text.setReadOnly(True)
        self.check_text.setMaximumHeight(170)
        root.addWidget(self.check_text)

    def _sync_spinner(self) -> None:
        if self._llm_model_busy or self._llm_inference_busy:
            self.llm_busy_spinner.start()
        else:
            self.llm_busy_spinner.stop()

    def reset_timing(self) -> None:
        self._llm_model_busy = False
        self._llm_inference_busy = False
        self.llm_model_time.setText("Model Load: -")
        self.llm_inference_time.setText("Inference: -")
        self._sync_spinner()

    def set_model_loading(self, elapsed_s: float) -> None:
        self._llm_model_busy = True
        self.llm_model_time.setText(f"Model Load: running {float(elapsed_s):.1f}s")
        self._sync_spinner()

    def set_model_loaded(self, elapsed_s: float, cache_hit: bool = False) -> None:
        self._llm_model_busy = False
        suffix = " (cache hit)" if cache_hit else ""
        self.llm_model_time.setText(f"Model Load: {float(elapsed_s):.2f}s{suffix}")
        self._sync_spinner()

    def set_model_load_failed(self, elapsed_s: float | None = None) -> None:
        self._llm_model_busy = False
        if elapsed_s is None:
            self.llm_model_time.setText("Model Load: failed")
        else:
            self.llm_model_time.setText(f"Model Load: failed after {float(elapsed_s):.2f}s")
        self._sync_spinner()

    def set_inference_running(self, elapsed_s: float) -> None:
        self._llm_inference_busy = True
        self.llm_inference_time.setText(f"Inference: running {float(elapsed_s):.1f}s")
        self._sync_spinner()

    def set_inference_done(self, elapsed_s: float) -> None:
        self._llm_inference_busy = False
        self.llm_inference_time.setText(f"Inference: {float(elapsed_s):.2f}s")
        self._sync_spinner()

    def set_inference_failed(self, elapsed_s: float | None = None) -> None:
        self._llm_inference_busy = False
        if elapsed_s is None:
            self.llm_inference_time.setText("Inference: failed")
        else:
            self.llm_inference_time.setText(f"Inference: failed after {float(elapsed_s):.2f}s")
        self._sync_spinner()

class FSAPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("fsa_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        title_row = QHBoxLayout()
        title_row.addWidget(TitleLabel("FSA+BFS+OPT", self))
        title_row.addStretch(1)
        self.fsa_busy_spinner = BusySpinner(self, size=16, color="#f2cc60")
        self.fsa_busy_time = CaptionLabel("", self)
        title_row.addWidget(self.fsa_busy_spinner)
        title_row.addWidget(self.fsa_busy_time)
        root.addLayout(title_row)
        root.addWidget(BodyLabel("FSA+BFS+OPT Reference - Connect", self))
        self.fsa_connect_table = QTableWidget(self)
        self.fsa_connect_table.setColumnCount(4)
        self.fsa_connect_table.setHorizontalHeaderLabels(["Step", "Operation", "Cost", "Duration"])
        fsa_connect_header = self.fsa_connect_table.horizontalHeader()
        fsa_connect_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        fsa_connect_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        fsa_connect_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        fsa_connect_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.fsa_connect_table.setAlternatingRowColors(True)
        root.addWidget(self.fsa_connect_table, 1)
        root.addWidget(BodyLabel("FSA+BFS+OPT Reference - Process", self))
        self.fsa_process_table = QTableWidget(self)
        self.fsa_process_table.setColumnCount(4)
        self.fsa_process_table.setHorizontalHeaderLabels(["Step", "Operation", "Cost", "Duration"])
        fsa_process_header = self.fsa_process_table.horizontalHeader()
        fsa_process_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        fsa_process_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        fsa_process_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        fsa_process_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.fsa_process_table.setAlternatingRowColors(True)
        root.addWidget(self.fsa_process_table, 2)

    def set_busy(self, busy: bool, elapsed_s: float | None = None) -> None:
        if busy:
            self.fsa_busy_spinner.start()
            t = 0.0 if elapsed_s is None else float(elapsed_s)
            self.fsa_busy_time.setText(f"running {t:.1f}s")
        else:
            self.fsa_busy_spinner.stop()
            if elapsed_s is None:
                self.fsa_busy_time.setText("")
            else:
                self.fsa_busy_time.setText(f"done {float(elapsed_s):.2f}s")

class LogPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("log_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.addWidget(TitleLabel("Log", self))
        root.addWidget(BodyLabel("Runtime Log", self))
        self.log_text = QPlainTextEdit(self)
        self.log_text.setReadOnly(True)
        root.addWidget(self.log_text, 1)

class SettingsPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("settings_page")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        root.addWidget(TitleLabel("Settings", self))
        root.addWidget(BodyLabel("Runtime options", self))

        adapter_card = CardWidget(self)
        adapter_layout = QGridLayout(adapter_card)
        adapter_layout.setContentsMargins(14, 12, 14, 12)
        adapter_layout.setHorizontalSpacing(10)
        adapter_layout.setVerticalSpacing(8)
        self.adapter_edit = QLineEdit(self)
        self.adapter_edit.setPlaceholderText("Select LoRA adapter directory")
        self.btn_adapter = PushButton("Browse Adapter", self)
        adapter_layout.addWidget(BodyLabel("Adapter Dir", self), 0, 0)
        adapter_layout.addWidget(self.adapter_edit, 0, 1)
        adapter_layout.addWidget(self.btn_adapter, 0, 2)
        self.prewarm_status = CaptionLabel("Model prewarm: idle", self)
        adapter_layout.addWidget(self.prewarm_status, 1, 1)
        adapter_layout.setColumnStretch(1, 1)
        root.addWidget(adapter_card)

        card = CardWidget(self)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        text_col.addWidget(BodyLabel("Persist temporary artifacts", self))
        text_col.addWidget(
            CaptionLabel(
                "Controls whether recipe XML/JSON and FSA+BFS+OPT LLM corpus JSONL are written to disk.",
                self,
            )
        )
        card_layout.addLayout(text_col, 1)

        self.persist_files_switch = SwitchButton(self)
        self.persist_files_switch.setOnText("On")
        self.persist_files_switch.setOffText("Off")
        self.persist_files_switch.setChecked(False)
        card_layout.addWidget(self.persist_files_switch, 0, Qt.AlignmentFlag.AlignRight)

        root.addWidget(card)

        llm_limit_card = CardWidget(self)
        llm_limit_layout = QGridLayout(llm_limit_card)
        llm_limit_layout.setContentsMargins(14, 12, 14, 12)
        llm_limit_layout.setHorizontalSpacing(10)
        llm_limit_layout.setVerticalSpacing(8)
        llm_limit_layout.addWidget(BodyLabel("LLM Limits", self), 0, 0)
        llm_limit_layout.addWidget(
            CaptionLabel("Single-run generation timeout.", self),
            0,
            1,
            1,
            2,
        )
        llm_limit_layout.addWidget(BodyLabel("Timeout (s)", self), 1, 0)
        self.llm_attempt_timeout_spin = QSpinBox(self)
        self.llm_attempt_timeout_spin.setRange(10, 3600)
        self.llm_attempt_timeout_spin.setValue(DEFAULT_LLM_ATTEMPT_TIMEOUT_S)
        llm_limit_layout.addWidget(self.llm_attempt_timeout_spin, 1, 1)
        llm_limit_layout.setColumnStretch(2, 1)
        root.addWidget(llm_limit_card)

        root.addStretch(1)

__all__ = [
    "FSAPage",
    "HomePage",
    "LLMPage",
    "LogPage",
    "SettingsPage",
]
