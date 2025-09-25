import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json


def load_4bit_model():
    if torch.cuda.is_available():
        print(f"✅ GPU พร้อมใช้งาน: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("❌ ไม่พบ GPU - จะใช้ CPU (ไม่แนะนำสำหรับ 4-bit)")

    local_model_path = "./models/gemma-3"

    print("🔄 กำลังโหลด tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,              
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_quant_type="nf4",    
        bnb_4bit_use_double_quant=True,   
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=quantization_config,
        device_map="auto",       
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True  
    )

    return model, tokenizer


def setup_lora_config():
    return LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=[ 
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1, 
        bias="none",      
        task_type="CAUSAL_LM" 
    )


def prepare_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def format_prompt(question, answer):
        return f"Question: {question}\nAnswer: {answer}<|endoftext|>"

    formatted_data = [{"text": format_prompt(item['question'], item['answer'])} for item in data]

    return Dataset.from_list(formatted_data)


def tokenize_dataset(dataset, tokenizer, max_length=512):

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        tokenized_output["labels"] = tokenized_output["input_ids"].clone()

        return tokenized_output

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def fine_tune_model(model, tokenizer, train_dataset, output_dir="./gemma3-finetuned"):

    model = prepare_model_for_kbit_training(model)

    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=3, 
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=4,  
        learning_rate=2e-4, 
        warmup_steps=100, 

        optim="paged_adamw_8bit", 
        gradient_checkpointing=True, 
        dataloader_pin_memory=False,

        save_steps=500,
        save_total_limit=2,
        logging_steps=50,

        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("🚀 เริ่ม Fine-tuning...")
    trainer.train()

    trainer.save_model()
    print(f"✅ บันทึกโมเดลเสร็จที่: {output_dir}")

    return model, trainer


if __name__ == "__main__":
    model, tokenizer = load_4bit_model()

    train_dataset = prepare_dataset("dataset.json")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)

    model, trainer = fine_tune_model(model, tokenizer, train_dataset)

    print(f"\n✅ Fine-tuning เสร็จสิ้น!")
