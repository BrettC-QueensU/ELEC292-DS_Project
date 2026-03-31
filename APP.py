import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QWidget,
    QGridLayout, QLabel, QFileDialog, QPushButton, QVBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


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


class ImageGallery(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Gallery")
        self.setGeometry(100, 100, 900, 700)
        self.thumbnail_size = 400
        self.columns = 2
        self._viewers = []

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)
        root_layout.addWidget(self.load_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

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


app = QApplication(sys.argv)
window = ImageGallery()
window.show()
sys.exit(app.exec_())