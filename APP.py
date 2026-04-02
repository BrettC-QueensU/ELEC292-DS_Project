import matplotlib.pyplot as plt
import pandas as pd
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QPushButton, QLabel, QScrollArea,
    QGridLayout, QFileDialog
)
from PyQt5.QtGui import (QPixmap, QFont)
from PyQt5.QtCore import Qt
from preprocessing import preprocess_csv


# ─── Tab 1: Home ────────────────────────────────────────────────────────────
class HomeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Welcome to the Home Tab!\n \nUpload a csv using the upload button to check running vs jumping accelerometer data")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Calibri", 16))

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setVisible(False)

        btn = QPushButton("Upload")
        btn.setFont(QFont("Calibri", 16))
        btn.clicked.connect(self.file_browse)

        layout.addWidget(label)
        layout.addWidget(self.image_label)
        layout.addStretch()
        layout.addWidget(btn)
        self.setLayout(layout)

    def file_browse(self):
        paths, _ = QFileDialog.getOpenFileName(
            self, "Select a csv file", "",
            "CSV Files (*.csv)"
        )
        if paths:
            self.dataset = pd.read_csv(paths)
            self.dataset = preprocess_csv(self.dataset)



            time = self.dataset['Time (s)']
            ax = self.dataset['Linear Acceleration x (m/s^2)']
            ay = self.dataset['Linear Acceleration y (m/s^2)']
            az = self.dataset['Linear Acceleration z (m/s^2)']

            fig, raw = plt.subplots(figsize=(12, 5))


            raw.plot(time, ax, label='X', color='tab:blue', linewidth=1.2)
            raw.plot(time, ay, label='Y', color='tab:orange', linewidth=1.2)
            raw.plot(time, az, label='Z', color='tab:green', linewidth=1.2)

            raw.set_xlabel('Time (s)', fontsize=12)
            raw.set_ylabel('Linear Acceleration (m/s²)', fontsize=12)
            raw.set_title('Linear Acceleration vs Time (X, Y, Z)', fontsize=14)
            raw.legend(title='Axis', fontsize=11)
            raw.grid(True, linestyle='--', alpha=0.5)
            raw.axhline(0, color='black', linewidth=0.8, linestyle='-')


            plt.tight_layout()
            plt.savefig('inputcsv.png', dpi=150)
            plt.close(fig)

            self.display_graph()

    def display_graph(self):
        pixmap = QPixmap('inputcsv.png')
        scaled = pixmap.scaledToWidth(
            self.width() - 40,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setVisible(True)

    def resizeEvent(self, event):
        if not self.image_label.isVisible():
            return
        self.display_graph()
        super().resizeEvent(event)

# ─── Tab 2: Image Gallery ────────────────────────────────────────────────────

class ImageViewer(QMainWindow):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(path)
        self.setGeometry(150, 150, 800, 600)

        self.original_pixmap = QPixmap(path)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: #1a1a1a;")
        self.setCentralWidget(self.label)

        self.update_image()

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def update_image(self):
        if self.original_pixmap.isNull():
            return
        scaled = self.original_pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)


class ImageGalleryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thumbnail_size = 400
        self.columns = 2
        self._viewers = []

        layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)
        layout.addWidget(self.load_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        scroll.setWidget(self.grid_widget)

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        if paths:
            self.populate_grid(paths)

    def populate_grid(self, paths):
        # Clear existing thumbnails
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

        for index, path in enumerate(paths):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue

            pixmap = pixmap.scaled(
                self.thumbnail_size, self.thumbnail_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(self.thumbnail_size, self.thumbnail_size)
            label.setStyleSheet("border: 1px solid #ccc; background: #1a1a1a;")
            label.mousePressEvent = lambda e, p=path: self.open_fullsize(p)

            row = index // self.columns
            col = index % self.columns
            self.grid_layout.addWidget(label, row, col)

    def open_fullsize(self, path):
        viewer = ImageViewer(path, parent=self)
        viewer.setAttribute(Qt.WA_DeleteOnClose)
        viewer.show()
        self._viewers.append(viewer)


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Tab App")
        self.setMinimumSize(1200, 800)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        self.tabs.addTab(HomeTab(),          "🏠 Home")
        self.tabs.addTab(ImageGalleryTab(),  "🖼️ Gallery")

        self.tabs.currentChanged.connect(self.on_tab_change)
        self.setCentralWidget(self.tabs)

    def on_tab_change(self, index):
        print(f"Switched to tab index: {index} — '{self.tabs.tabText(index)}'")

    def go_to_tab(self, index):
        self.tabs.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())