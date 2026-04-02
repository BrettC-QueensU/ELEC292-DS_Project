import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import learning_curve
from featureExtraction import extract_features

HDF5_PATH = './hdf5_data.h5'
MODEL_PATH = './trained_model.pkl'

ACC_COLS = [
    'Linear Acceleration x (m/s^2)',
    'Linear Acceleration y (m/s^2)',
    'Linear Acceleration z (m/s^2)',
    'Absolute acceleration (m/s^2)',
]

def load_windows_from_hdf5(hdf5_path, split='Train'):
    """
    Reads every window stored under /Segmented data/<split>/
    Returns (X, y) where X is (n_windows, n_features) and y is 0=walking, 1=jumping.
    """
    x_list, y_list = [], []
    label_map = {'walking': 0, 'jumping': 1}

    with h5py.File(hdf5_path, 'r') as f:
        split_group = f[f'Segmented data/{split}']

        for label_name, label_int in label_map.items():
            if label_name not in split_group:
                print(f'  [WARNING] label "{label_name}" not found in {split} split — skipping.')
                continue

            label_group = split_group[label_name]
            for window_key in label_group:
                raw = label_group[window_key][:]  # numpy array

                # Shape handling – expect (n_samples, 4 or 5)
                if raw.ndim == 2 and raw.shape[1] == 5:
                    raw = raw[:, :4]          # drop time column if present
                elif raw.ndim == 2 and raw.shape[0] == 5:
                    raw = raw[:4, :]

                if raw.ndim == 2 and raw.shape[0] == 4:
                    raw = raw.T               # → (n_samples, 4)

                # DataFrame with ACC_COLS as index so extract_features can use .loc
                window_df = pd.DataFrame(
                    raw.T,                    # shape (4, n_samples)
                    index=ACC_COLS,
                )

                feat_row = extract_features(window_df)   # (1, 40)
                x_list.append(feat_row.values[0])
                y_list.append(label_int)

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return x, y


def train_and_save_model():
    """Train the logistic regression pipeline and persist it to MODEL_PATH."""
    print('Loading training windows …')
    x_train, y_train = load_windows_from_hdf5(HDF5_PATH, split='Train')
    x_test,  y_test  = load_windows_from_hdf5(HDF5_PATH, split='Test')

    # Pipeline: StandardScaler fitted on training data only (no leakage)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, random_state=42),
    )

    clf.fit(x_train, y_train)
    print('\nModel trained successfully.')

    # Evaluate
    y_test_pred = clf.predict(x_test)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f'\n--- Classification Results ---')
    print(f'  Test accuracy : {test_acc:.4f}  ({test_acc * 100:.1f} %)')
    print(f'  Test recall   : {test_recall:.4f}')

    # Learning curves
    print('\nComputing learning curves …')
    train_sizes, train_scores, val_scores = learning_curve(
        estimator   = clf,
        X           = x_train,
        y           = y_train,
        train_sizes = np.linspace(0.1, 1.0, 10),
        cv          = 5,
        scoring     = 'accuracy',
        n_jobs      = -1,
        shuffle     = True,
        random_state= 42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Step 6 – Logistic Regression Classification Results', fontsize=14)

    # Learning Curves
    ax = axes[0]
    ax.set_title('Learning Curves')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Accuracy')
    ax.plot(train_sizes, train_mean, 'o-',  color='royalblue',  label='Training accuracy')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='royalblue')
    ax.plot(train_sizes, val_mean,   's--', color='darkorange', label='CV accuracy (validation)')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='darkorange')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Chance level (50 %)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Confusion Matrix
    ax = axes[1]
    ax.set_title(f'Confusion Matrix (Test set)\nAccuracy = {test_acc * 100:.1f} %')
    cm   = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jumping'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    # ROC Curve
    ax = axes[2]
    ax.set_title('ROC Curve (Test set)')
    y_test_prob = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    auc_score   = roc_auc_score(y_test, y_test_prob)
    print(f'AUC : {auc_score:.4f}')
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score)
    disp.plot(ax=ax, curve_kwargs={'color': 'royalblue'})
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.004, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('step6_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Save model ─────────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    print(f'\nModel saved to "{MODEL_PATH}"')

    return clf


if __name__ == '__main__':
    train_and_save_model()
