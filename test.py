import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Define label mapping
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# ✅ Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Prediction function using local image file
def predict_tflite_from_path(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output and interpret result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output_data)
    pred_label = label_map.get(pred_idx, "Unknown")
    
    print(f"✅ Predicted Label: {pred_label} (Index: {pred_idx})")

# ✅ Example usage (change this path to your uploaded/test image)
predict_tflite_from_path("image.jpg")
