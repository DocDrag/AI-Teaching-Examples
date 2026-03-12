import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-3-1b-it"
OUTPUT_DIR = "./gemma3-finetuned"


def prepare_dataset(tokenizer):
    print("🔄 กำลังเตรียม Dataset...")
    # โหลดไฟล์ dataset.json
    dataset = load_dataset('json', data_files='dataset.json', split='train')

    # ฟังก์ชันแปลงคำถาม-คำตอบ ให้เป็น Prompt ข้อความเดียวที่ AI เข้าใจ
    def format_and_tokenize(example):
        text = f"คำถาม: {example['question']}\nตอบ: {example['answer']}{tokenizer.eos_token}"
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    return tokenized_dataset


def main():
    if torch.cuda.is_available():
        print(f"✅ GPU พร้อมใช้งาน: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ไม่พบ GPU - การเทรน 4-bit อาจทำงานได้ไม่สมบูรณ์")

    print("🔄 กำลังโหลด tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("🔄 กำลังโหลด Base Model แบบ 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # เตรียม Dataset
    train_dataset = prepare_dataset(tokenizer)

    # เตรียม Model สำหรับ Fine-tune
    model = prepare_model_for_kbit_training(model)

    # ตั้งค่า LoRA Config (เพิ่ม target_modules ที่ครอบคลุม)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,  # ปรับจำนวนรอบตามความเหมาะสม
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        save_steps=50,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 เริ่ม Fine-tuning...")
    trainer.train()

    print(f"💾 กำลังบันทึกโมเดล (Adapter) ไปที่ {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("✅ เสร็จสิ้นกระบวนการ Fine-tuning!")


if __name__ == "__main__":
    main()