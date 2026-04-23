import os
import sys
import site

# Try to resolve macOS Anaconda "cocoa plugin not found" issue
try:
    site_packages = site.getsitepackages()[0]
    plugin_path = os.path.join(site_packages, "PyQt6", "Qt6", "plugins")
    platforms_path = os.path.join(plugin_path, "platforms")
    if os.path.exists(platforms_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms_path
        os.environ["QT_PLUGIN_PATH"] = plugin_path
except Exception:
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont
from vqa_engine import VQAEngine

# QThread for background VQA task to keep UI responsive
class VQATaskThread(QThread):
    status_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, image_path, question):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.question = question

    def run(self):
        try:
            self.status_update.emit("Processing Image+Question with Phi-3.5-vision...This might take a moment if weights are loading.")
            answer = self.engine.get_answer_vision(self.image_path, self.question)
            
            self.finished.emit(answer)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Document VQA (Phi-3.5-Vision)")
        self.setMinimumSize(900, 700)
        
        self.engine = VQAEngine()
        self.current_image_path = None
        self.vqa_thread = None

        self._init_ui()
        self._check_dependencies()

    def _init_ui(self):
        # Apply a clean, polished light stylesheet
        style_sheet = """
            QMainWindow {
                background-color: #F8FAFC;
            }
            QWidget {
                background-color: #F8FAFC;
                color: #0F172A;
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
            QPushButton:disabled {
                background-color: #E2E8F0;
                color: #94A3B8;
            }
            QPushButton#RemoveBtn {
                background-color: #EF4444;
            }
            QPushButton#RemoveBtn:hover {
                background-color: #DC2626;
            }
            QLabel#Title {
                font-size: 28px;
                font-weight: 700;
                color: #0F172A;
                margin-bottom: 20px;
                letter-spacing: 0.5px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 12px;
                background-color: #FFFFFF;
                color: #0F172A;
                font-size: 15px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #2563EB;
                background-color: #FFFFFF;
            }
            QLabel#ImagePlaceholder {
                border: 2px dashed #94A3B8;
                border-radius: 12px;
                background-color: #F1F5F9;
                color: #64748B;
                font-size: 16px;
                font-weight: 500;
            }
            QLabel#StatusLabel {
                color: #475569;
                font-weight: 500;
            }
        """
        self.setStyleSheet(style_sheet)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Document VQA System (Phi-3.5-Vision Local)")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Image Upload Section
        upload_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Document Image")
        self.upload_btn.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_btn)
        
        self.remove_btn = QPushButton("Remove Image")
        self.remove_btn.setObjectName("RemoveBtn")
        self.remove_btn.clicked.connect(self.remove_image)
        self.remove_btn.hide()  # Hidden until an image is loaded
        upload_layout.addWidget(self.remove_btn)
        
        upload_layout.addStretch()

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setWordWrap(True)
        upload_layout.addWidget(self.status_label)
        upload_layout.setStretch(0, 0)
        upload_layout.setStretch(1, 0)
        upload_layout.setStretch(2, 1) # stretch the spacer
        upload_layout.setStretch(3, 2) # give status label space
        
        main_layout.addLayout(upload_layout)

        # Image Display Area
        self.image_label = QLabel("No image selected")
        self.image_label.setObjectName("ImagePlaceholder")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(350)
        self.image_label.setScaledContents(True)
        main_layout.addWidget(self.image_label, stretch=1)

        # Question Section
        question_layout = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about the document... e.g. 'What is the total amount?'")
        self.question_input.returnPressed.connect(self.submit_question)
        question_layout.addWidget(self.question_input)

        self.submit_btn = QPushButton("Ask")
        self.submit_btn.clicked.connect(self.submit_question)
        self.submit_btn.setEnabled(False)  # Disable until image is uploaded
        question_layout.addWidget(self.submit_btn)

        main_layout.addLayout(question_layout)

        # Answer Section
        self.answer_area = QTextEdit()
        self.answer_area.setReadOnly(True)
        self.answer_area.setPlaceholderText("The generated answer will appear here...")
        self.answer_area.setMaximumHeight(150)
        main_layout.addWidget(self.answer_area)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate mode
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

    def _check_dependencies(self):
        success, msg = self.engine.check_dependencies()
        if not success:
            QMessageBox.warning(self, "Dependency Warning", msg)
            self.status_label.setText("Dependency missing. Check warnings.")

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.current_image_path = file_name
            # Display image preserving aspect ratio
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("border: 1px solid #CBD5E1; border-radius: 12px;") 
            self.submit_btn.setEnabled(True)
            self.remove_btn.show()
            self.status_label.setText("Image loaded. You can now ask questions.")

    def remove_image(self):
        self.current_image_path = None
        self.image_label.clear()
        self.image_label.setText("No image selected")
        self.image_label.setStyleSheet("")
        self.submit_btn.setEnabled(False)
        self.remove_btn.hide()
        self.status_label.setText("Image removed.")
        self.answer_area.clear()

    def resizeEvent(self, event):
        # Update image scaling on resize if an image is loaded
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def submit_question(self):
        question = self.question_input.text().strip()
        if not self.current_image_path:
            QMessageBox.information(self, "No Image", "Please upload a document image first.")
            return
        if not question:
            QMessageBox.information(self, "No Question", "Please enter a question.")
            return

        # Prepare UI for processing
        self.submit_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.question_input.setEnabled(False)
        self.answer_area.clear()
        self.progress_bar.show()

        # Start background thread
        self.vqa_thread = VQATaskThread(self.engine, self.current_image_path, question)
        self.vqa_thread.status_update.connect(self.update_status)
        self.vqa_thread.finished.connect(self.on_vqa_finished)
        self.vqa_thread.error.connect(self.on_vqa_error)
        self.vqa_thread.start()

    def update_status(self, text):
        self.status_label.setText(text)

    def on_vqa_finished(self, answer):
        self.answer_area.setPlainText(answer)
        self._reset_ui_state()
        self.status_label.setText("Answer ready.")

    def on_vqa_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during VQA process:\n{error_msg}")
        self.answer_area.setPlainText("Failed to generate answer.")
        self._reset_ui_state()
        self.status_label.setText("Error occurred.")

    def _reset_ui_state(self):
        self.progress_bar.hide()
        self.submit_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.question_input.setEnabled(True)
        self.question_input.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
