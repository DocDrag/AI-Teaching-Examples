import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download, login


def download_and_load_4bit_model():
    """ดาวน์โหลดและโหลดโมเดล gemma-3 แบบ 4-bit"""

    load_dotenv()  # โหลดไฟล์ .env
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("ไม่พบ HF_TOKEN กรุณาตั้งค่าใน environment หรือ .env file")
    login(token=hf_token)

    if torch.cuda.is_available():
        print(f"✅ GPU พร้อมใช้งาน: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("❌ ไม่พบ GPU - จะใช้ CPU (ไม่แนะนำสำหรับ 4-bit)")

    local_model_path = "./models/gemma-3"

    print("🔄 กำลังดาวน์โหลดโมเดล...")
    print("⚠️  หาก error ให้ตรวจสอบว่าได้รับ access แล้วที่:")
    print("   https://huggingface.co/google/gemma-3-1b-it")

    snapshot_download(
        repo_id="google/gemma-3-1b-it",
        local_dir=local_model_path,
        resume_download=True
    )
    print("✅ ดาวน์โหลดเสร็จ")

    print("🔄 กำลังโหลด tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ ตั้งค่า pad_token เรียบร้อย")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
    )

    print("🔄 กำลังโหลดโมเดลแบบ 4-bit...")
    print("⚠️  ขั้นตอนนี้อาจใช้เวลา 1-2 นาที...")

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=quantization_config,
        device_map="auto",  
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True 
    )

    print("✅ โหลดโมเดล 4-bit เสร็จ!")

    return local_model_path

if __name__ == "__main__":
    model_path = download_and_load_4bit_model()

    print("\n🎉 เสร็จสิ้น!")
    print(f"📁 โมเดลเก็บอยู่ที่: {model_path}")