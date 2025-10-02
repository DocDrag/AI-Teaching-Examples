import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import fitz 
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, glob, os

# -------------------- Config --------------------
EMB_MODEL = "intfloat/multilingual-e5-base"
MODEL_ID = "google/gemma-3-1b-it"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
PDF_PATH_PATTERN = "./docs/*.pdf"


# -------------------- Utils --------------------
def read_pdf(path):
    """อ่านข้อความจาก PDF"""
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pages.append(page.get_text("text").strip())
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """แบ่งข้อความเป็น chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# -------------------- Embeddings --------------------
class SimpleIndex:
    def __init__(self, model_name=EMB_MODEL):
        self.embedder = SentenceTransformer(model_name)
        self.chunks = []
        self.embs = None

    def build(self, texts):
        self.chunks = texts
        self.embs = self.embedder.encode(
            ["passage: " + t for t in texts],
            normalize_embeddings=True
        )

    def search(self, query, top_k=3):
        q_emb = self.embedder.encode(["query: " + query], normalize_embeddings=True)
        scores = np.dot(self.embs, q_emb[0])
        idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], scores[i]) for i in idx]


# -------------------- LLM --------------------
class SimpleLLM:
    def __init__(self, model_id=MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=300):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# -------------------- Pipeline --------------------
if __name__ == "__main__":
    # 1) โหลด PDF ทุกไฟล์
    pdf_files = sorted(glob.glob(PDF_PATH_PATTERN))
    if not pdf_files:
        raise SystemExit("❌ ไม่พบไฟล์ PDF ในโฟลเดอร์ docs/")

    all_text = ""
    for path in pdf_files:
        print(f"📄 กำลังอ่าน {os.path.basename(path)}")
        pages = read_pdf(path)
        all_text += "\n".join(pages) + "\n"

    # 2) ตัดเป็น chunks
    chunks = chunk_text(all_text)

    # 3) สร้างดัชนี
    index = SimpleIndex()
    index.build(chunks)

    # 4) สร้าง LLM
    llm = SimpleLLM()

    # 5) คำถาม
    question = input("❓ คำถามที่อยากถามจากเอกสาร: ")

    results = index.search(question, top_k=3)
    context = "\n\n".join([r[0] for r in results])

    prompt = f"""
    คุณเป็นผู้ช่วยที่ตอบจากเอกสาร PDF เท่านั้น
    ห้ามคัดลอกคำถามหรือข้อมูลมาเปล่า ๆ ต้องตอบเป็นคำตอบชัดเจน

    คำถาม: {question}

    ข้อมูลที่เกี่ยวข้อง:
    {context}
    
    (โปรดตอบเป็นภาษาไทย ชัดเจนและเข้าใจง่าย)
    คำตอบ:
    """

    answer = llm.generate(prompt)
    # เอาเฉพาะบรรทัดแรก
    answer = answer.strip().split("คำตอบ:")[1]

    print("\n===== SUMMARY =====\n")
    print(answer)
