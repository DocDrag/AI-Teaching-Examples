import pandas as pd

# โหลดข้อมูลลูกค้า
print("กำลังโหลดข้อมูลลูกค้า...")
dataset = pd.read_csv("customer.csv", low_memory=False)

print(f"ข้อมูลทั้งหมด: {len(dataset)} รายการ")
print(f"อลัมน์ในข้อมูล: {list(dataset.columns)}")

# เลือกเฉพาะคอลัมน์ที่ต้องการใช้งาน
data = dataset[['Recency', 'Frequency', 'Monetary', 'CustomerID']]

# แปลงค่าที่ไม่สามารถแปลงเป็นตัวเลขได้ให้กลายเป็น NaN
data[['Recency', 'Frequency', 'Monetary']] = data[['Recency', 'Frequency', 'Monetary']].apply(
    pd.to_numeric, errors='coerce'
)

# แสดงข้อมูลก่อนทำความสะอาด
print(f"\n  ก่อนทำความสะอาดข้อมูล:")
print(f"   - ข้อมูลที่ไม่สมบูรณ์: {data.isnull().sum().sum()} ตัว")
print(f"   - แถวที่สมบูรณ์: {len(data.dropna())} จาก {len(data)} แถว")

# ลบแถวที่มีค่าว่าง (เฉพาะ Recency/Frequency/Monetary)
data = data.dropna(subset=['Recency', 'Frequency', 'Monetary'])

print(f" หลังทำความสะอาด: เหลือ {len(data)} รายการสมบูรณ์")

# -------------------------------
# บันทึกเป็นไฟล์ใหม่
# -------------------------------
output_file = "customer_clean_data.csv"
data.to_csv(output_file, index=False)

print(f"💾 บันทึกข้อมูลที่ทำความสะอาดแล้วไว้ที่: {output_file}")
