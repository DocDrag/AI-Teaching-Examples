import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("กำลังโหลดข้อมูลที่ทำความสะอาดแล้ว...")
df = pd.read_csv("customer_clean_data.csv")

print("ตัวอย่างข้อมูลก่อนการปรับสเกล:")
print(df.head())

# เลือกเฉพาะคอลัมน์ตัวเลข
features = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# แสดงตัวอย่าง
print("ตัวอย่างข้อมูลหลังการสเกล:")
print(df_scaled.head())

# บันทึกข้อมูลและ scaler
df_scaled.to_csv("customer_scaled_data.csv", index=False)
joblib.dump(scaler, "customer_scaler.pkl")

print("\nบันทึกแล้ว:")
print(" - customer_scaled_data.csv")
print(" - customer_scaler.pkl")
