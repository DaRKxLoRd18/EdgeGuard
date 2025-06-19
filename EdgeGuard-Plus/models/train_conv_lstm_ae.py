import os, numpy as np, tensorflow as tf
from glob import glob
from sklearn.metrics import roc_auc_score
from conv_lstm_ae import build_conv_lstm_ae
import matplotlib.pyplot as plt

# Load only train (normal sequences)
def load_train(dir):
    arr = [np.load(f) for f in glob(os.path.join(dir, "Train*.npy"))]
    return np.concatenate(arr)

# Reconstruct threshold based on validation split
def train():
    X_train = load_train("data/processed/train")
    X_train = X_train.astype('float32')

    # Split val from train
    val_split = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:val_split], X_train[val_split:]

    model = build_conv_lstm_ae()
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_tr, X_tr,
                        validation_data=(X_val, X_val),
                        batch_size=16, epochs=30,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    model.save("saved_model/conv_lstm_ae.h5")

    # Determine threshold from validation
    recon_val = model.predict(X_val)
    errs = np.mean((X_val - recon_val)**2, axis=(1,2,3,4))
    threshold = np.percentile(errs, 95)
    print("Threshold:", threshold)

    # Evaluate on test
    X_test = np.concatenate([np.load(f) for f in glob("data/processed/test/Test*.npy")])
    y_true = np.concatenate([np.load(f) for f in glob("data/processed/test/Test*_labels.npy")])
    recon_test = model.predict(X_test)
    errs_test = np.mean((X_test - recon_test)**2, axis=(1,2,3,4))
    y_pred = (errs_test > threshold).astype(int)

    auc = roc_auc_score(y_true, errs_test)
    print(f"AUC: {auc:.3f}")

    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("ae_loss.png")

if __name__ == "__main__":
    train()