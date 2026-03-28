import sys
import pandas as pd
import pickle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from preprocessing import preprocess_csv


def run_classification(input_path):
    """
    Load the input CSV, classify each window, and return an output DataFrame.
    Replace the body of this function with your classification logic.
    """

    # preprocess data
    df_processed = preprocess_csv(input_path)

    # extract features

    # load the logistic regression model
    with open('logistic_model.pkl', 'wb') as file:
        model = pickle.load(file)

    # run the model


    output_df = pd.DataFrame({'label': []})  # placeholder
    return output_df


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ELEC 292 – Activity Classifier')
        self.setMinimumSize(500, 220)

        self.input_path = None
        self.output_df  = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 12)
        layout.setSpacing(12)

        # title
        title = QLabel('Walking / Jumping Classifier')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # input file row
        input_row = QHBoxLayout()
        self.btn_load   = QPushButton('Load Input CSV')
        self.label_file = QLabel('No file selected')
        self.label_file.setStyleSheet('color: #666666;')
        input_row.addWidget(self.btn_load)
        input_row.addWidget(self.label_file, stretch=1)
        layout.addLayout(input_row)

        # run button
        self.btn_run = QPushButton('Run Classification')
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet('background-color: #1565C0; color: white; border-radius: 4px;')
        layout.addWidget(self.btn_run)

        # save button
        self.btn_save = QPushButton('Save Output CSV')
        self.btn_save.setMinimumHeight(36)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet('background-color: #2e7d32; color: white; border-radius: 4px;')
        layout.addWidget(self.btn_save)

        # status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage('Ready')

        # connect signals
        self.btn_load.clicked.connect(self.load_csv)
        self.btn_run.clicked.connect(self.run)
        self.btn_save.clicked.connect(self.save_csv)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Accelerometer CSV', '', 'CSV Files (*.csv)')
        if not path:
            return
        self.input_path = path
        self.label_file.setText(path)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.status.showMessage('CSV loaded – press Run Classification')

    def run(self):
        self.status.showMessage('Running …')
        QApplication.processEvents()
        try:
            self.output_df = run_classification(self.input_path)
            self.btn_save.setEnabled(True)
            self.status.showMessage('Classification complete – press Save Output CSV')
        except Exception as e:
            self.status.showMessage(f'Error: {e}')

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Output CSV', 'output_labels.csv', 'CSV Files (*.csv)')
        if not path:
            return
        self.output_df.to_csv(path, index=False)
        self.status.showMessage(f'Saved → {path}')


app = QApplication(sys.argv)
app.setStyle('Fusion')
window = MainWindow()
window.show()
sys.exit(app.exec_())