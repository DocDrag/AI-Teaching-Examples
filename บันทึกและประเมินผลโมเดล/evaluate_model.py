import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-1b-it"
FINETUNED_ADAPTER_PATH = "./gemma3-finetuned"

print("🔄 กำลังโหลดโมเดลเพื่อประเมินผล...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
model.eval()

# ชุดคำถามสำหรับประเมินผล (Test Set)
test_questions = [
    "มะเขือเทศพันธุ์ San Marzano มีลักษณะเด่นอะไร",  # คำถามที่มีใน Data
    "ทำไมใบมะเขือเทศถึงม้วนเข้าด้านใน",  # คำถามที่มีใน Data
    "ปลูกมะเขือเทศใช้ดินแบบไหนดี",  # คำถามดัดแปลง (คล้ายใน Data)
    "แตงโมปลูกยังไงคะ?"  # คำถามหลอก (นอกเรื่อง)
]

print("\n📊 เริ่มการประเมินโมเดล (Model Evaluation)...\n")
for q in test_questions:
    prompt = f"คำถาม: {q}\nตอบ:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    print(f"❓ คำถาม: {q}")
    print(f"🤖 คำตอบ: {answer}\n" + "-" * 30)