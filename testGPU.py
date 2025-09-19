import tensorflow as tf
import torch

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU name: {gpu.name} ({torch.cuda.get_device_name(0)})")
else:
    print("❌ ไม่พบ GPU หรือ cuDNN ยังไม่ได้เชื่อมต่อกับ TensorFlow")

# ตรวจสอบ CUDA runtime version
print("CUDA runtime version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())