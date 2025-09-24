import os, pickle, numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# โหลดโมเดลตรวจจับ + สร้าง embedding
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

refs = {}
for person in os.listdir("ref_faces"):
    person_dir = os.path.join("ref_faces", person)
    if not os.path.isdir(person_dir): continue

    embeddings = []
    for file in os.listdir(person_dir):
        if file.endswith((".jpg", ".png")):
            img = Image.open(os.path.join(person_dir, file)).convert("RGB")
            face = mtcnn(img)
            if face is not None:
                emb = resnet(face.unsqueeze(0)).detach().numpy()
                embeddings.append(emb)

    if embeddings:
        refs[person] = np.mean(embeddings, axis=0)

# เซฟโมเดล
with open("face_model.pkl", "wb") as f:
    pickle.dump(refs, f)
print("บันทึกโมเดลเสร็จสิ้น!")
