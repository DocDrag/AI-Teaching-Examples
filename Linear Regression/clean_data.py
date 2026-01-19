import pandas as pd

# โหลดข้อมูลสภาพอากาศต่างประเทศปี 1940 - 1945
print("กำลังโหลดข้อมูลสภาพอากาศ...")
dataset = pd.read_csv("Weather.csv", low_memory=False)

print(f"ข้อมูลทั้งหมด: {len(dataset)} วัน")
print(f"คอลัมน์ในข้อมูล: {list(dataset.columns)}")

# เลือกเฉพาะคอลัมน์ที่ต้องการใช้งาน
data = dataset[['MaxTemp', 'MinTemp', 'Precip', 'MeanTemp']]

# แปลงค่าที่ไม่สามารถแปลงเป็นตัวเลขได้ให้กลายเป็น NaN
data = data.apply(pd.to_numeric, errors='coerce')

# แสดงข้อมูลก่อนทำความสะอาด
print(f"\nก่อนทำความสะอาดข้อมูล:")
print(f"   - ข้อมูลที่ไม่สมบูรณ์: {data.isnull().sum().sum()} ตัว")
print(f"   - แถวที่สมบูรณ์: {len(data.dropna())} จาก {len(data)} แถว")

# ลบแถวที่มีค่าว่าง
data = data.dropna()

print(f"หลังทำความสะอาด: เหลือ {len(data)} วันที่สมบูรณ์")

# -------------------------------
# บันทึกเป็นไฟล์ใหม่
# -------------------------------
output_file = "Weather_clean_data.csv"
data.to_csv(output_file, index=False)

print(f"บันทึกข้อมูลที่ทำความสะอาดแล้วไว้ที่: {output_file}")
