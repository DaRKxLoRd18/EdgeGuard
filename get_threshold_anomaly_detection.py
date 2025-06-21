'''
Will Run on kaggle

'''


import numpy as np
from glob import glob
from sklearn.metrics import f1_score, roc_auc_score

def load_data(test_dir):
    X_test = np.concatenate([np.load(f) for f in glob(f"{test_dir}/Test[0-9][0-9][0-9].npy")])
    y_true = np.concatenate([np.load(f) for f in glob(f"{test_dir}/Test[0-9][0-9][0-9]_labels.npy")])
    return X_test, y_true


def compute_errors(model, X):
    X_pred = model.predict(X)
    return np.mean((X - X_pred) ** 2, axis=(1, 2, 3, 4))

def find_best_threshold(y_true, errors, method='f1'):
    thresholds = np.linspace(min(errors), max(errors), 1000)
    best_thresh = 0
    best_score = -1

    for t in thresholds:
        y_pred = (errors > t).astype(int)
        score = f1_score(y_true, y_pred) if method == 'f1' else roc_auc_score(y_true, errors)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score

def percentile_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)

# --- Main script ---

from tensorflow.keras.models import load_model

model = load_model("saved_model/final_conv_lstm_ae.h5", compile=False)
X_test, y_true = load_data("data/processed/test")
errors = compute_errors(model, X_test)

# 1. Best F1 score threshold
best_f1_thresh, best_f1 = find_best_threshold(y_true, errors, method='f1')
print(f"📌 Best F1 Threshold: {best_f1_thresh:.8f}  |  F1 Score: {best_f1:.4f}")

# 2. 95th percentile threshold
perc_thresh = percentile_threshold(errors, 95)
print(f"📌 95th Percentile Threshold: {perc_thresh:.8f}")

# 3. AUC (just for monitoring quality)
auc = roc_auc_score(y_true, errors)
print(f"📊 AUC Score: {auc:.4f}")
