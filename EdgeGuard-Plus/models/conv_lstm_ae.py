# EdgeGuard-Plus/models/conv_lstm_ae.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D

def build_conv_lstm_ae(seq_len=12, height=64, width=64, channels=1):
    inp = Input(shape=(seq_len, height, width, channels))

    # --- Encoder ---
    x = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',
                   return_sequences=True, activation='relu')(inp)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same',
                   return_sequences=True, activation='relu')(x)
    x = BatchNormalization()(x)

    # --- Decoder ---
    x = ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same',
                   return_sequences=True, activation='relu')(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',
                   return_sequences=True, activation='relu')(x)
    x = BatchNormalization()(x)

    out = Conv3D(filters=1, kernel_size=(3,3,3), activation='sigmoid', padding='same')(x)

    return Model(inputs=inp, outputs=out)
