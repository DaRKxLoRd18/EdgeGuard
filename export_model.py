import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np

# Paths
h5_model_path = "saved_model/final_conv_lstm_ae.h5"
onnx_model_path = "saved_model/final_conv_lstm_ae.onnx"
tflite_model_path = "saved_model/final_conv_lstm_ae.tflite"

# Load model
model = tf.keras.models.load_model(h5_model_path, compile=False)

# Input signature for ONNX export (batch, 12, 64, 64, 1)
spec = (tf.TensorSpec((None, 12, 64, 64, 1), tf.float32, name="input"),)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)
print("✅ Exported to:", onnx_model_path)

# Convert to TFLite with Select TF Ops
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,        # enable standard ops
    tf.lite.OpsSet.SELECT_TF_OPS           # enable TF ops (ConvLSTM needs this)
]
converter._experimental_lower_tensor_list_ops = False  # crucial!
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Exported to:", tflite_model_path)
