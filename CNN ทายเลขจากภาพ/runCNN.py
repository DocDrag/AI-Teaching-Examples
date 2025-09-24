import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2

# โหลดโมเดล
model = tf.keras.models.load_model('cnn_mnist_augmented_model.keras')

def predict_digit(image_path):
    # อ่านภาพเป็น grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # ถ้าตัวเลขเป็นดำบนพื้นขาว → กลับสี
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Threshold + หา bounding box
    _, binary = cv2.threshold(img_array, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit_img = img_array[y:y+h, x:x+w]
    else:
        digit_img = img_array

    # Resize + normalize
    digit_pil = Image.fromarray(digit_img).resize((28,28), Image.Resampling.LANCZOS)
    img_ready = np.array(digit_pil).astype('float32') / 255.0
    img_ready = img_ready.reshape(1,28,28,1)

    # ทำนาย
    pred = model.predict(img_ready, verbose=0)[0]
    return np.argmax(pred), np.max(pred)

# ใช้งาน
num, conf = predict_digit("my_digit_test.png")
print(f"ทำนายเป็น {num} (ความมั่นใจ {conf:.2%})")
