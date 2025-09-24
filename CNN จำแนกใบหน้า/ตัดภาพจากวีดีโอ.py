import cv2
import os
from facenet_pytorch import MTCNN
from PIL import Image

VIDEO_PATH = "videos/name.mp4"   # วิดีโอคนนี้
SAVE_DIR = "ref_faces/name"
os.makedirs(SAVE_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # เอาเฉพาะทุก ๆ 5 เฟรม
    if frame_count % 5 != 0:
        continue

    # ตรวจจับใบหน้า
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            face_img = frame[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("RGB")
            save_path = os.path.join(SAVE_DIR, f"frame_{frame_count}_{i}.jpg")
            pil_img.save(save_path)
            save_count += 1

cap.release()
print(f"บันทึกใบหน้าทั้งหมด {save_count} ภาพ")
