import cv2
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# โหลดตัวตรวจจับและโมเดล face embedding
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

# โหลด reference (ที่สร้างไว้จาก train.py)
with open("face_model.pkl", "rb") as f:
    refs = pickle.load(f)

def cosine_sim(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-10)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            face = img.crop((x1,y1,x2,y2))
            face_tensor = mtcnn(face)
            if face_tensor is None: continue

            emb = resnet(face_tensor.unsqueeze(0)).detach().numpy()
            # หาคนที่คล้ายที่สุด
            best_name = "Unknown"
            best_score = 0.0
            for n, ref in refs.items():
                sim = cosine_sim(emb, ref)
                if sim > best_score:
                    best_name = n
                    best_score = sim

            if best_score >= 0.6:
                final_name = best_name
                color = (0, 255, 0)  # สีเขียว (รู้จัก)
            else:
                final_name = "Unknown"
                color = (0, 0, 255)  # สีแดง (ไม่รู้จัก/ไม่มั่นใจ)

                # วาดกรอบและชื่อ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # แสดงชื่อพร้อมคะแนนความมั่นใจ
            score = float(best_score) * 100
            text_display = f"{final_name} ({score:.2f}%)"
            cv2.putText(frame, text_display, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break

cap.release()
cv2.destroyAllWindows()
