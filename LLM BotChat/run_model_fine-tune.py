import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL_ID = "google/gemma-3-1b-it"
FINETUNED_ADAPTER_PATH = "./gemma3-finetuned"

print("🔄 กำลังโหลด Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🔄 กำลังโหลด Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("🔄 กำลังประกอบร่างกับข้อมูล Fine-tune (Adapter)...")
model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ ใช้งานบน: {device}")


def clean_response(text, original_prompt):
    if text.startswith(original_prompt):
        text = text[len(original_prompt):].strip()
    # ตัดข้อความส่วนเกินที่เกิดจากตัวจบประโยคของ Gemma
    if tokenizer.eos_token in text:
        text = text.split(tokenizer.eos_token)[0].strip()
    return text


def generate_text(prompt, max_tokens=150, temperature=0.5, top_p=0.9):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_text = clean_response(generated_text, prompt)

        return cleaned_text if cleaned_text else "ขออภัยฉันไม่เข้าใจคำถาม"

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🤖 Gemma-3 (Fine-tuned Bot) พร้อมใช้งาน!")
    print("=" * 50)

    while True:
        user_input = input("\n💭 You: ")

        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            print("❌ กรุณาพิมพ์ข้อความ")
            continue

        # ใช้ Prompt รูปแบบเดียวกับตอนที่เราสั่งเทรน
        formatted_prompt = f"คำถาม: {user_input}\nตอบ:"
        response = generate_text(formatted_prompt)
        print(f"🤖 Bot: {response}")