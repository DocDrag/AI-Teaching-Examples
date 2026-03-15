import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_PATH = "./gemma3-finetuned"
SAVE_PATH = "./gemma3-bot-final"

print("🔄 กำลังโหลด Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# โหลด Base Model แบบความละเอียดปกติ (16-bit) เพื่อเตรียมรวมร่าง
print("🔄 กำลังโหลด Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu" # ใช้ CPU ในการรวมร่างเพื่อป้องกัน VRAM เต็ม
)

print("🔄 กำลังสวมแว่นตาความรู้ (Adapter)...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("✨ กำลังรวมร่างสมองหลักเข้ากับความรู้ใหม่ (Merge and Unload)...")
model = model.merge_and_unload()

print(f"💾 กำลังบันทึกโมเดลพร้อมใช้งานไปที่: {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("✅ เสร็จสิ้น! โมเดลของคุณพร้อมสำหรับนำไปใช้จริงแล้ว")