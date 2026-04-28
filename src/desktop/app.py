import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.pipeline import load_item_order, process_all_images
from src.core.review import ReviewItem, find_review_items
from src.item_matcher import ItemMatcher
from src.json_updater import update_owned_materials
from src.count_ocr_backend import CountOcrBackend, build_backend


def _cv2_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR image to QPixmap."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class AnalyzeWorker(QThread):
    """Background thread for OCR processing."""
    finished = Signal(dict, dict)
    error = Signal(str)

    def __init__(self, images, item_order, matcher, reader):
        super().__init__()
        self.images = images
        self.item_order = item_order
        self.matcher = matcher
        self.reader = reader

    def run(self):
        try:
            results, cell_images = process_all_images(
                self.images, self.item_order, self.matcher, self.reader
            )
            self.finished.emit(results, cell_images)
        except Exception as e:
            self.error.emit(str(e))


class InputPage(QWidget):
    analyze_requested = Signal(str, list)

    def __init__(self):
        super().__init__()
        self._images: list[np.ndarray] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("1. Justin Planner JSON"))
        self.json_input = QTextEdit()
        self.json_input.setPlaceholderText("justin163.json 내용을 붙여넣으세요...")
        self.json_input.setMaximumHeight(150)
        layout.addWidget(self.json_input)

        layout.addWidget(QLabel("2. Screenshots (Ctrl+V)"))
        self.paste_label = QLabel("Ctrl+V로 스크린샷을 붙여넣으세요")
        self.paste_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.paste_label.setStyleSheet(
            "border: 2px dashed #555; border-radius: 8px; padding: 30px; color: #888;"
        )
        layout.addWidget(self.paste_label)

        self.thumb_layout = QHBoxLayout()
        self.thumb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(self.thumb_layout)

        self.btn_analyze = QPushButton("분석 시작")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._on_analyze)
        layout.addWidget(self.btn_analyze)

        self.loading_label = QLabel("분석 중...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

        layout.addStretch()

    def paste_image(self):
        clipboard = QApplication.clipboard()
        img = clipboard.image()
        if img.isNull():
            return

        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        bpl = img.bytesPerLine()  # may be > w*3 due to 4-byte row alignment
        ptr = img.bits()
        arr = np.array(ptr, dtype=np.uint8).reshape(h, bpl)
        arr = arr[:, :w * 3].reshape(h, w, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._images.append(bgr)
        self._update_thumbnails()
        self._update_button()

    def _update_thumbnails(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, img in enumerate(self._images):
            thumb = QLabel()
            pix = _cv2_to_qpixmap(img)
            thumb.setPixmap(pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio))
            self.thumb_layout.addWidget(thumb)

        count = len(self._images)
        if count > 0:
            self.paste_label.setText(f"{count}장 추가됨 (Ctrl+V로 더 추가)")
        else:
            self.paste_label.setText("Ctrl+V로 스크린샷을 붙여넣으세요")

    def _update_button(self):
        has_json = bool(self.json_input.toPlainText().strip())
        has_images = len(self._images) > 0
        self.btn_analyze.setEnabled(has_json and has_images)

    def _on_analyze(self):
        json_text = self.json_input.toPlainText().strip()
        self.analyze_requested.emit(json_text, self._images.copy())


class ReviewPage(QWidget):
    review_completed = Signal(dict)

    def __init__(self):
        super().__init__()
        self._items: list[ReviewItem] = []
        self._index = 0
        self._reviewed: dict[str, int] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 18px; color: #aaa;")
        layout.addWidget(self.progress_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self.mid_label = QLabel()
        self.mid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mid_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.mid_label)

        self.qty_input = QLineEdit()
        self.qty_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qty_input.setStyleSheet("font-size: 24px; font-weight: bold; max-width: 200px;")
        self.qty_input.setMaximumWidth(200)
        self.qty_input.returnPressed.connect(self._confirm)
        layout.addWidget(self.qty_input, alignment=Qt.AlignmentFlag.AlignCenter)

        self.reasons_label = QLabel()
        self.reasons_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reasons_label.setStyleSheet("color: #e74c3c; font-size: 13px;")
        layout.addWidget(self.reasons_label)

        actions = QHBoxLayout()
        actions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Enter: 확정")
        hint.setStyleSheet("color: #666; font-size: 13px;")
        actions.addWidget(hint)
        btn_skip = QPushButton("Skip")
        btn_skip.clicked.connect(self._skip)
        actions.addWidget(btn_skip)
        layout.addLayout(actions)

        layout.addStretch()

    def set_items(self, items: list[ReviewItem], name_map: dict[str, str] | None = None):
        self._items = items
        self._index = 0
        self._reviewed = {}
        self._name_map = name_map or {}
        self._show_current()

    def _show_current(self):
        item = self._items[self._index]
        self.progress_label.setText(f"{self._index + 1} / {len(self._items)}")

        pix = _cv2_to_qpixmap(item.cell_image)
        scaled = pix.scaled(
            300, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.image_label.setPixmap(scaled)

        display_name = self._name_map.get(item.material_id, item.material_id)
        self.mid_label.setText(display_name)
        self.qty_input.setText(str(item.ocr_qty))
        self.reasons_label.setText(", ".join(item.reasons))
        self.qty_input.setFocus()
        self.qty_input.selectAll()

    def _confirm(self):
        try:
            val = int(self.qty_input.text())
        except ValueError:
            self.qty_input.selectAll()
            return
        self._reviewed[self._items[self._index].material_id] = val
        self._advance()

    def _skip(self):
        item = self._items[self._index]
        self._reviewed[item.material_id] = item.ocr_qty
        self._advance()

    def _advance(self):
        self._index += 1
        if self._index < len(self._items):
            self._show_current()
        else:
            self.review_completed.emit(self._reviewed)


class ResultPage(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("결과"))

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.result_text)

        self.btn_copy = QPushButton("복사")
        self.btn_copy.clicked.connect(self._copy)
        layout.addWidget(self.btn_copy)

    def set_result(self, json_text: str):
        self.result_text.setPlainText(json_text)

    def _copy(self):
        QApplication.clipboard().setText(self.result_text.toPlainText())
        self.btn_copy.setText("복사 완료!")
        QTimer.singleShot(2000, lambda: self.btn_copy.setText("복사"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blue Archive Inventory Analyzer")
        self.setMinimumSize(500, 600)

        self._json_text = ""
        self._confirmed: dict[str, int] = {}
        self._worker: AnalyzeWorker | None = None
        self._justin_data: dict = {}

        self._reader: CountOcrBackend | None = None
        self._matcher: ItemMatcher | None = None
        self._item_order: list[str | None] | None = None
        self._name_map: dict[str, str] = {}

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.input_page = InputPage()
        self.review_page = ReviewPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.input_page)
        self.stack.addWidget(self.review_page)
        self.stack.addWidget(self.result_page)

        self.input_page.analyze_requested.connect(self._on_analyze)
        self.review_page.review_completed.connect(self._on_review_done)

        # Intercept Ctrl+V on all child widgets so image paste takes priority
        self.input_page.json_input.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Intercept Ctrl+V globally: if clipboard has an image, paste as screenshot."""
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_V
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            and self.stack.currentWidget() is self.input_page
        ):
            clipboard = QApplication.clipboard()
            mime = clipboard.mimeData()
            if mime and mime.hasImage():
                self.input_page.paste_image()
                return True  # consumed — don't let QTextEdit paste it as text
        return super().eventFilter(obj, event)

    def _get_reader(self) -> CountOcrBackend:
        if self._reader is None:
            self._reader = build_backend()
        return self._reader

    def _get_matcher(self) -> ItemMatcher:
        if self._matcher is None:
            from src.runtime_path import data_path
            self._matcher = ItemMatcher(data_path("references"))
        return self._matcher

    def _get_item_order(self) -> list[str | None]:
        if self._item_order is None:
            from src.runtime_path import data_path
            self._item_order, self._name_map = load_item_order(data_path("item_order.json"))
        return self._item_order

    def _on_analyze(self, json_text: str, images: list[np.ndarray]):
        try:
            self._justin_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Error", f"Invalid JSON: {e}")
            return

        self._json_text = json_text
        self.input_page.loading_label.show()
        self.input_page.btn_analyze.setEnabled(False)

        self._worker = AnalyzeWorker(
            images,
            self._get_item_order(),
            self._get_matcher(),
            self._get_reader(),
        )
        self._worker.finished.connect(self._on_analyze_done)
        self._worker.error.connect(self._on_analyze_error)
        self._worker.start()

    def _on_analyze_done(self, results: dict, cell_images: dict):
        self.input_page.loading_label.hide()

        if not results:
            QMessageBox.warning(self, "Error", "No items detected")
            self.input_page.btn_analyze.setEnabled(True)
            return

        existing = self._justin_data.get("owned_materials", {})
        review_items = find_review_items(results, existing, cell_images, threshold=0.9)

        review_mids = {item.material_id for item in review_items}
        self._confirmed = {mid: qty for mid, (qty, _) in results.items() if mid not in review_mids}

        if review_items:
            self.review_page.set_items(review_items, self._name_map)
            self.stack.setCurrentWidget(self.review_page)
        else:
            self._finalize({})

    def _on_analyze_error(self, msg: str):
        self.input_page.loading_label.hide()
        self.input_page.btn_analyze.setEnabled(True)
        QMessageBox.warning(self, "Error", msg)

    def _on_review_done(self, reviewed: dict[str, int]):
        self._finalize(reviewed)

    def _finalize(self, reviewed: dict[str, int]):
        all_updates = {}
        for mid, qty in self._confirmed.items():
            all_updates[mid] = qty
        for mid, qty in reviewed.items():
            all_updates[mid] = qty

        update_owned_materials(self._justin_data, all_updates)
        result_text = json.dumps(self._justin_data, indent=2, ensure_ascii=False)
        self.result_page.set_result(result_text)
        self.stack.setCurrentWidget(self.result_page)


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
