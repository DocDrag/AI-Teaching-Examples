import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_model_path = "./gemma3-finetuned"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้งานบน: {device}")

if not torch.cuda.is_available():
    model.to(device)


def clean_response(text, original_prompt):
    # ลบ prompt ออกจากต้นข้อความ
    if text.startswith(original_prompt):
        text = text[len(original_prompt):].strip()

    # ตัดที่ <|endoftext|> ตัวจบประโยค
    if '<|endoftext|>' in text:
        text = text.split('<|endoftext|>')[0].strip()

    return text


def generate_text(prompt, max_tokens=50, temperature=0.7, top_p=0.9):
    try:
        formatted_prompt = f"คำถาม: {prompt}\nตอบ: "

        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
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
                num_return_sequences=1,
                early_stopping=True, 
                no_repeat_ngram_size=3, 
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        cleaned_text = clean_response(generated_text, formatted_prompt)

        if cleaned_text.startswith("ตอบ: "):
            cleaned_text = cleaned_text[5:].strip()

        return cleaned_text if cleaned_text else "ขออภัยฉันไม่เข้าใจคำถาม"

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


def chat_with_model():
    print("\n" + "=" * 50)
    print("🤖 Gemma-3 Chat Bot พร้อมใช้งาน!")
    print("=" * 50)

    while True:
        user_input = input("\n💭 You: ")

        if not user_input.strip():
            print("❌ กรุณาพิมพ์ข้อความ")
            continue

        print("🤔 กำลังคิด...")
        response = generate_text(
            user_input,
            max_tokens=50,
            temperature=0.8,
            top_p=0.9
        )
        print(f"🤖 AI: {response}")


if __name__ == "__main__":
    chat_with_model()