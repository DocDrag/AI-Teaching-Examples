import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# โหลดข้อมูล MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape เป็น (batch, height, width, channel) และ Normalize
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# แปลง label เป็น one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# สร้างโมเดล CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== เพิ่ม Data Augmentation =====
datagen = ImageDataGenerator(
    rotation_range=10,          # หมุน ±10 องศา
    width_shift_range=0.1,      # เลื่อนซ้าย-ขวา 10%
    height_shift_range=0.1,     # เลื่อนขึ้น-ลง 10%
    zoom_range=0.1,             # ซูม ±10%
    fill_mode='constant',       # เติมพื้นที่ว่างด้วยสีดำ
    cval=0                      # ค่าสีที่เติม (0=ดำ)
)

# ฝึกโมเดลด้วย Data Augmentation
print("Training with Data Augmentation...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=15,  # เพิ่ม epochs เพราะ data หลากหลายขึ้น
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ประเมินผลโมเดล
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# บันทึกโมเดลใหม่
model_path = 'cnn_mnist_augmented_model.keras'
model.save(model_path)
print(f"Improved model saved to {model_path}")