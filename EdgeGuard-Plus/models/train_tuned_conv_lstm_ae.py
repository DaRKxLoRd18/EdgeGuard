'''
Note : This is demo code
This code will be run in Kaggle environment



'''





import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D

# Load dataset
def load_data(train_dir, test_dir):
    X_train = np.concatenate([np.load(f) for f in glob(os.path.join(train_dir, "Train*.npy"))])
    y_test = np.concatenate([np.load(f) for f in sorted(glob(os.path.join(test_dir, "Test*_labels.npy")))])
    X_test = np.concatenate([
        np.load(f)
        for f in sorted(glob(os.path.join(test_dir, "Test[0-9][0-9][0-9].npy")))
    ])
    return X_train.astype('float32'), X_test.astype('float32'), y_test.astype('int32')

# Model builder for KerasTuner
def build_model(hp):
    inp = Input(shape=(12, 64, 64, 1))

    # Encoder
    x = ConvLSTM2D(
        filters=hp.Choice("filters_1", [16, 32, 64]),
        kernel_size=hp.Choice("kernel_1", [3, 5]),
        padding="same",
        return_sequences=True,
        activation="relu"
    )(inp)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(
        filters=hp.Choice("filters_2", [32, 64]),
        kernel_size=hp.Choice("kernel_2", [3, 5]),
        padding="same",
        return_sequences=True,
        activation="relu"
    )(x)
    x = BatchNormalization()(x)

    # Decoder
    x = ConvLSTM2D(
        filters=hp.Choice("filters_3", [32, 64]),
        kernel_size=hp.Choice("kernel_3", [3, 5]),
        padding="same",
        return_sequences=True,
        activation="relu"
    )(x)
    x = BatchNormalization()(x)

    out = Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice("lr", [1e-3, 1e-4])),
        loss="mse"
    )

    return model

# Main function
def train():
    BASE = "/kaggle/input/ucsd-ped2-processed/data/processed"
    X_train, X_test, y_test = load_data(BASE + "/train", BASE + "/test")

    # Train/val split
    split = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split], X_train[split:]

    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='convlstm_ae'
    )

    tuner.search_space_summary()

    tuner.search(
        X_tr, X_tr,
        validation_data=(X_val, X_val),
        epochs=30,
        batch_size=16,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint("best_tuned_model.h5", save_best_only=True, monitor="val_loss")
        ]
    )

    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("🔍 Best Hyperparameters:")
    for k, v in best_hp.values.items():
        print(f"{k}: {v}")

    # Final train plot
    recon_val = best_model.predict(X_val)
    errs_val = np.mean((X_val - recon_val) ** 2, axis=(1, 2, 3, 4))
    threshold = np.percentile(errs_val, 95)
    print("🔑 Threshold:", threshold)

    recon_test = best_model.predict(X_test)
    errs_test = np.mean((X_test - recon_test) ** 2, axis=(1, 2, 3, 4))
    y_pred = (errs_test > threshold).astype(int)

    auc = roc_auc_score(y_test, errs_test)
    print("🎯 AUC:", round(auc, 3))
    print("📊 F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save reconstruction loss plot
    plt.plot(tuner.oracle.get_best_trials(num_trials=1)[0].metrics.get_history('val_loss'), label='val_loss')
    plt.title("Best Trial Val Loss")
    plt.legend()
    plt.savefig("best_trial_val_loss.png")

if __name__ == "__main__":
    train()
