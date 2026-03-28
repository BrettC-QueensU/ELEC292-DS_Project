import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import learning_curve


# ── Load features and labels from the HDF5 file ───────────────────────────────
with h5py.File('dataset.h5', 'r') as f:
    X_train     = f['segmented_data/train/features'][:]
    y_train_raw = f['segmented_data/train/labels'][:]
    X_test      = f['segmented_data/test/features'][:]
    y_test_raw  = f['segmented_data/test/labels'][:]

# HDF5 stores strings as bytes - decode if necessary
y_train_raw = np.array([lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl) for lbl in y_train_raw])
y_test_raw  = np.array([lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl) for lbl in y_test_raw])


# ── Encode string labels into integers ────────────────────────────────────────
# e.g. "jumping" --> 0, "walking" --> 1
le = LabelEncoder()
le.fit(np.concatenate([y_train_raw, y_test_raw]))
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)


# ── Define and train the classifier pipeline ──────────────────────────────────
# StandardScaler normalizes the features, LogisticRegression is the classifier
# Using make_pipeline ensures the scaler is only fit on training data,
# preventing data leakage into the test set
l_reg = LogisticRegression(max_iter=10000)
clf   = make_pipeline(StandardScaler(), l_reg)

clf.fit(X_train, y_train)


# ── Obtaining the predictions and the probabilities ───────────────────────────
y_pred     = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)  # y_clf_prob[:, 1] is the probability of the positive class for each sample


# ── Obtaining the classification accuracy and recall ─────────────────────────
acc       = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, clf.predict(X_train))
recall    = recall_score(y_test, y_pred)

print(f'train accuracy is: {train_acc:.4f}')
print(f'accuracy is: {acc:.4f}')
print(f'recall is: {recall:.4f}')


# ── Plotting the confusion matrix ─────────────────────────────────────────────
cm         = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=le.classes_)

fig, ax = plt.subplots()
cm_display.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.show()


# ── Plotting the ROC curve ────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display  = RocCurveDisplay(fpr=fpr, tpr=tpr)

fig, ax = plt.subplots()
roc_display.plot(ax=ax)
ax.set_title('ROC Curve - Test Set')
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.show()

auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print(f'the AUC is: {auc:.4f}')


# ── Learning curves ───────────────────────────────────────────────────────────
train_sizes, train_scores, val_scores = learning_curve(
    estimator   = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
    X           = X_train,
    y           = y_train,
    train_sizes = np.linspace(0.1, 1.0, 10),
    cv          = 5,
    scoring     = 'accuracy',
    n_jobs      = -1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_mean, 'o-', color='royalblue',  label='Training accuracy')
ax.plot(train_sizes, val_mean,   's-', color='darkorange', label='CV accuracy')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='royalblue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkorange')
ax.axhline(y=acc, linestyle='--', color='green', label=f'Final test accuracy ({acc * 100:.1f} %)')
ax.set_xlabel('Training set size (windows)')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curves - Logistic Regression')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/learning_curves.png', dpi=150)
plt.show()


# ── Save the trained model and label encoder ──────────────────────────────────
# Saved as a pickle file so the desktop app (app.py) can load it
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'label_encoder': le}, f)

print('Model saved --> logistic_model.pkl')