def clean_dataset(input_file="dataset.txt", output_file="cleaned_dataset.txt"):
    valid_labels = {"positive", "negative", "neutral"}
    seen = set()  # เก็บไว้เช็คข้อมูลซ้ำ
    cleaned_lines = []

    # ตัวนับสถิติ
    total = 0
    removed_invalid_format = 0
    removed_invalid_label = 0
    removed_empty_sentence = 0
    removed_duplicates = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                total -= 1 # ไม่นับรวมช่องว่าง
                continue  # ข้ามบรรทัดว่าง

            if "<LABEL>" not in line:
                removed_invalid_format += 1
                continue

            parts = line.split("<LABEL>")
            if len(parts) != 2:
                removed_invalid_format += 1
                continue

            sentence, label = parts[0].strip(), parts[1].strip()

            if label not in valid_labels:
                removed_invalid_label += 1
                continue

            if not sentence:
                removed_empty_sentence += 1
                continue

            key = (sentence, label)
            if key in seen:
                removed_duplicates += 1
                continue

            seen.add(key)
            cleaned_lines.append(f"{sentence}<LABEL>{label}")

    # เขียนไฟล์ cleaned dataset
    with open(output_file, "w", encoding="utf-8") as fout:
        for line in cleaned_lines:
            fout.write(line + "\n")

    # รายงานผล
    print(f"✅ ทำความสะอาดไฟล์ {input_file} → {output_file} เรียบร้อยแล้ว")
    print("📊 สถิติการลบ:")
    print(f"   - ทั้งหมดเดิม: {total}")
    print(f"   - ลบเพราะรูปแบบไม่ถูกต้อง: {removed_invalid_format}")
    print(f"   - ลบเพราะ label ไม่ถูกต้อง: {removed_invalid_label}")
    print(f"   - ลบเพราะประโยคว่าง: {removed_empty_sentence}")
    print(f"   - ลบเพราะซ้ำ: {removed_duplicates}")
    print(f"✅ เหลือทั้งหมด: {len(cleaned_lines)} ตัวอย่าง")


if __name__ == "__main__":
    clean_dataset()
