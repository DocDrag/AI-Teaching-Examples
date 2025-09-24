import deepcut

def tokenize_text(text: str) -> list[str]:
    """
    ตัดคำด้วย deepcut และลบช่องว่างเกินออก
    """
    text = text.strip()
    tokens = deepcut.tokenize(text)
    # ลบ token ที่เป็นช่องว่างหรือว่างเปล่า
    tokens = [t for t in tokens if t.strip()]
    return tokens


def main():
    input_file = "cleaned_dataset.txt"           # ไฟล์ข้อความดิบ
    output_word = "word.txt"             # ไฟล์ tokenized สำหรับ Word2Vec
    output_dataset = "dataset_format.txt"  # ไฟล์ tokenized พร้อม label

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_word, "w", encoding="utf-8") as fword, \
         open(output_dataset, "w", encoding="utf-8") as fdataset:

        for line in fin:
            line = line.strip()
            if not line:
                continue  # ข้ามบรรทัดว่าง

            # แยกประโยคกับ label ออก
            parts = line.split("<LABEL>")
            sentence = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else ""

            # ตัดคำ
            tokens = tokenize_text(sentence)

            # เขียนลง word.txt (ไม่มี label)
            fword.write(" ".join(tokens) + "\n")

            # เขียนลง dataset_format.txt (มี label)
            if label:
                fdataset.write(" ".join(tokens) + " <LABEL> " + label + "\n")
            else:
                fdataset.write(" ".join(tokens) + "\n")

    print(f"✅ แปลงไฟล์ {input_file} → {output_word}, {output_dataset} เรียบร้อยแล้ว")


if __name__ == '__main__':
    main()
