#import matplotlib
#matplotlib.use('Agg')   # non-interactive backend (safe before any Qt import)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import sys
import os
import joblib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGridLayout, QFileDialog, QMessageBox,
    QSizePolicy, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt

from preprocessing import preprocess_csv
from featureExtraction import extract_features

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_PATH = './trained_model.pkl'

ACC_COLS = [
    'Linear Acceleration x (m/s^2)',
    'Linear Acceleration y (m/s^2)',
    'Linear Acceleration z (m/s^2)',
    'Absolute acceleration (m/s^2)',
]

WINDOW_COLORS = {
    'walking': '#AED6F1',   # soft blue
    'jumping': '#F1948A',   # soft red
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def segment_dataframe(df):
    """Split a preprocessed DataFrame into ~5-second windows.

    Returns
    -------
    windows : list[DataFrame]   each window as a DataFrame
    samples_per_window : int
    """
    times = df['Time (s)'].values
    sample_period = float(np.median(np.diff(times)))
    samples_per_window = max(1, int(round(5.0 / sample_period)))

    n_complete = len(df) // samples_per_window
    windows = []
    for i in range(n_complete):
        start = i * samples_per_window
        end   = start + samples_per_window
        windows.append(df.iloc[start:end].reset_index(drop=True))
    return windows, samples_per_window


def load_model():
    """Load the pre-trained sklearn pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Please run logisticRegression.py first to train and save the model."
        )
    return joblib.load(MODEL_PATH)


def classify_windows(windows, clf):
    """Run feature extraction + classification on a list of window DataFrames."""
    labels = []
    for window in windows:
        # extract_features expects a DataFrame with ACC_COLS as the *index*
        # and one column per sample  → shape (4, n_samples)
        window_t = window[ACC_COLS].T.copy()
        window_t.index = ACC_COLS
        feat = extract_features(window_t)   # returns (1, n_features) DataFrame
        pred = clf.predict(feat.values)[0]
        labels.append('jumping' if pred == 1 else 'walking')
    return labels


def build_labeled_csv(processed_df, windows, labels):
    """Return a DataFrame that contains only the windowed rows with label columns."""
    rows = []
    for i, (window, label) in enumerate(zip(windows, labels)):
        w = window.copy()
        w.insert(0, 'window', i)
        w['label'] = label
        rows.append(w)
    return pd.concat(rows, ignore_index=True)


def make_comparison_plot(raw_df, processed_df, windows, labels, save_path='classification_result.png'):
    """
    Two-panel figure:
      Top    – raw accelerometer signal
      Bottom – preprocessed signal
    Both panels have coloured bands per window (blue=walking, red=jumping).
    """
    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Accelerometer Data — Raw vs Pre-processed  |  Window Classification',
                 fontsize=13, fontweight='bold', y=1.01)

    axis_styles = [
        ('Linear Acceleration x (m/s^2)', 'X', 'tab:blue'),
        ('Linear Acceleration y (m/s^2)', 'Y', 'tab:orange'),
        ('Linear Acceleration z (m/s^2)', 'Z', 'tab:green'),
    ]

    # ── Raw panel ────────────────────────────────────────────────────────
    t_raw = raw_df['Time (s)']
    for col, lbl, clr in axis_styles:
        ax_raw.plot(t_raw, raw_df[col], label=lbl, color=clr, linewidth=1.0, alpha=0.85)
    ax_raw.set_ylabel('Acceleration (m/s²)', fontsize=10)
    ax_raw.set_title('Raw Data', fontsize=11)
    ax_raw.axhline(0, color='black', linewidth=0.7, linestyle='-')
    ax_raw.grid(True, linestyle='--', alpha=0.4)
    ax_raw.legend(title='Axis', fontsize=9, loc='upper right')

    # ── Preprocessed panel ───────────────────────────────────────────────
    t_proc = processed_df['Time (s)']
    for col, lbl, clr in axis_styles:
        ax_proc.plot(t_proc, processed_df[col], label=lbl, color=clr, linewidth=1.0, alpha=0.85)
    ax_proc.set_xlabel('Time (s)', fontsize=10)
    ax_proc.set_ylabel('Acceleration (m/s²)', fontsize=10)
    ax_proc.set_title('Pre-processed Data (SMA filtered)', fontsize=11)
    ax_proc.axhline(0, color='black', linewidth=0.7, linestyle='-')
    ax_proc.grid(True, linestyle='--', alpha=0.4)
    ax_proc.legend(title='Axis', fontsize=9, loc='upper right')

    # ── Window bands on both panels ──────────────────────────────────────
    for ax in (ax_raw, ax_proc):
        y_min, y_max = ax.get_ylim()
        for i, (window, label) in enumerate(zip(windows, labels)):
            t0 = window['Time (s)'].iloc[0]
            t1 = window['Time (s)'].iloc[-1]
            color = WINDOW_COLORS[label]
            ax.axvspan(t0, t1, alpha=0.30, color=color, zorder=0)

            # Small label tag at the top of each band
            t_mid = (t0 + t1) / 2
            tag = 'W' if label == 'walking' else 'J'
            ax.text(t_mid, y_max, tag,
                    ha='center', va='top', fontsize=7.5,
                    color='#333333', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc=color,
                              ec='none', alpha=0.6))

    # ── Shared legend for window classes ─────────────────────────────────
    walk_patch = mpatches.Patch(facecolor=WINDOW_COLORS['walking'], alpha=0.5,
                                edgecolor='none', label='Walking (W)')
    jump_patch = mpatches.Patch(facecolor=WINDOW_COLORS['jumping'],  alpha=0.5,
                                edgecolor='none', label='Jumping (J)')
    fig.legend(handles=[walk_patch, jump_patch],
               loc='upper right', fontsize=9,
               framealpha=0.85, title='Window label')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Tab 1: Home ─────────────────────────────────────────────────────────────

class HomeTab(QWidget):
    def __init__(self):
        super().__init__()
        self._output_df    = None   # labeled DataFrame ready for download
        self._result_image = 'classification_result.png'

        # ── Load model once ──────────────────────────────────────────────
        self._clf = None
        try:
            self._clf = load_model()
        except FileNotFoundError as e:
            pass   # show error to user only when they actually upload a file

        # ── Layout ──────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Title / instructions
        self._status_label = QLabel(
            "Welcome!\n\n"
            "Upload a CSV file from your accelerometer to classify each\n"
            "5-second window as  Walking  or  Jumping."
        )
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setFont(QFont("Calibri", 14))
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # Result summary (hidden until classification)
        self._summary_label = QLabel()
        self._summary_label.setAlignment(Qt.AlignCenter)
        self._summary_label.setFont(QFont("Calibri", 12))
        self._summary_label.setVisible(False)
        root.addWidget(self._summary_label)

        # Scrollable image area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setVisible(False)
        scroll.setWidget(self._image_label)
        root.addWidget(scroll, stretch=1)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._upload_btn = QPushButton("📂  Upload CSV")
        self._upload_btn.setFont(QFont("Calibri", 13))
        self._upload_btn.setMinimumHeight(40)
        self._upload_btn.clicked.connect(self._on_upload)
        btn_row.addWidget(self._upload_btn)

        self._download_btn = QPushButton("💾  Download Labeled CSV")
        self._download_btn.setFont(QFont("Calibri", 13))
        self._download_btn.setMinimumHeight(40)
        self._download_btn.setVisible(False)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        root.addLayout(btn_row)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select accelerometer CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        # Guard: model must be loaded
        if self._clf is None:
            try:
                self._clf = load_model()
            except FileNotFoundError as e:
                QMessageBox.critical(self, "Model not found", str(e))
                return

        # ── Pipeline ─────────────────────────────────────────────────────
        try:
            raw_df = pd.read_csv(path)

            # 1. Pre-process
            processed_df = preprocess_csv(raw_df.copy())

            # 2. Segment into ~5-second windows
            windows, _ = segment_dataframe(processed_df)
            if len(windows) == 0:
                QMessageBox.warning(
                    self, "Too short",
                    "The CSV file is too short to form even one 5-second window.\n"
                    "Please upload a longer recording."
                )
                return

            # 3. Classify
            labels = classify_windows(windows, self._clf)

            # 4. Build labeled output DataFrame
            self._output_df = build_labeled_csv(processed_df, windows, labels)

            # 5. Plot
            make_comparison_plot(raw_df, processed_df, windows, labels, self._result_image)

            # 6. Update UI
            n_walk = labels.count('walking')
            n_jump = labels.count('jumping')
            self._summary_label.setText(
                f"✅  {len(labels)} windows classified  —  "
                f"🚶 Walking: {n_walk}   |   "
                f"🤸 Jumping: {n_jump}"
            )
            self._summary_label.setVisible(True)
            self._download_btn.setVisible(True)
            self._status_label.setText(
                f"File: {os.path.basename(path)}\n"
                "Blue bands = Walking  |  Red bands = Jumping"
            )
            self._show_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error during classification", str(e))
            raise

    def _on_download(self):
        if self._output_df is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save labeled CSV", "labeled_output.csv", "CSV Files (*.csv)"
        )
        if save_path:
            self._output_df.to_csv(save_path, index=False)
            QMessageBox.information(
                self, "Saved",
                f"Labeled CSV saved to:\n{save_path}"
            )

    def _show_plot(self):
        pixmap = QPixmap(self._result_image)
        if pixmap.isNull():
            return
        # Scale to available width (leaving room for margins)
        available_w = max(self.width() - 60, 200)
        scaled = pixmap.scaledToWidth(available_w, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)
        self._image_label.setVisible(True)

    def resizeEvent(self, event):
        if self._image_label.isVisible():
            self._show_plot()
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
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().deleteLater()

        for index, path in enumerate(paths):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            pixmap = pixmap.scaled(
                self.thumbnail_size, self.thumbnail_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        self.setWindowTitle("Walking / Jumping Classifier")
        self.setMinimumSize(1200, 800)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.addTab(HomeTab(),         "🏠  Home")
        self.tabs.addTab(ImageGalleryTab(), "🖼️  Gallery")
        self.tabs.currentChanged.connect(self._on_tab_change)
        self.setCentralWidget(self.tabs)

    def _on_tab_change(self, index):
        print(f"Switched to tab {index} — '{self.tabs.tabText(index)}'")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())