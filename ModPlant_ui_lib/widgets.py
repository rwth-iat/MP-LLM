from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QWidget

class BusySpinner(QFrame):
    def __init__(self, parent: QWidget | None = None, size: int = 16, color: str = "#58a6ff"):
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(45)
        self.setFixedSize(size, size)
        self.hide()

    def _on_tick(self) -> None:
        self._angle = (self._angle + 24) % 360
        self.update()

    def start(self) -> None:
        self.show()
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not self.isVisible():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(self._color)
        pen.setWidth(2)
        p.setPen(pen)
        r = self.rect().adjusted(2, 2, -2, -2)
        p.drawArc(r, (90 - self._angle) * 16, 240 * 16)
        p.end()
