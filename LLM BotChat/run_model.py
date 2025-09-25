import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("กำลังโหลดโมเดล...")

local_model_path = snapshot_download(
    repo_id="google/gemma-3-1b-it",
    local_dir="./models/gemma-3",
    resume_download=True
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
    device_map="auto" if torch.cuda.is_available() else None, 
    trust_remote_code=True 
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้งานบน: {device}")

if not torch.cuda.is_available():
    model.to(device)


def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        new_text = generated_text[len(prompt):].strip()
        return new_text

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


def chat_with_model():
    print("\n" + "=" * 50)
    print("🤖 gemma-3 Chat Bot พร้อมใช้งาน!")
    print("=" * 50)

    while True:
        user_input = input("\n💭 You: ")

        if not user_input.strip():
            print("❌ กรุณาพิมพ์ข้อความ")
            continue

        print("🤔 กำลังคิด...")
        response = generate_text(
            user_input,
            max_length=150,
            temperature=0.8,
            top_p=0.9
        )
        print(f"🤖 AI: {response}")


if __name__ == "__main__":
    chat_with_model()