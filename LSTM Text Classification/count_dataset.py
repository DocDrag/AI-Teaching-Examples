def count_labels(dataset_path: str):
    # ตัวนับจำนวน label
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "<LABEL>" not in line:
                continue  # ข้ามถ้า format ไม่ถูกต้อง

            try:
                _, label = line.split("<LABEL>")
                label = label.strip().lower()
                if label in counts:
                    counts[label] += 1
            except ValueError:
                continue  # ข้ามถ้า split ไม่ได้

    return counts


if __name__ == "__main__":
    dataset_path = "cleaned_dataset.txt"
    result = count_labels(dataset_path)
    print("จำนวนตัวอย่างใน dataset:")
    for label, count in result.items():
        print(f"{label}: {count}")
