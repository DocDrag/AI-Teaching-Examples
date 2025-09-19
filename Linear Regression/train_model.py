import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# โหลดข้อมูล
print("🌤️  กำลังโหลดข้อมูลสภาพอากาศที่ทำความสะอาดแล้ว...")
dataset = pd.read_csv("Weather_clean_data.csv", low_memory=False)

# เลือกคอลัมน์ที่ต้องการ
data = dataset[['MaxTemp', 'MinTemp', 'Precip', 'MeanTemp']].copy()

# สร้าง target ใหม่: MeanTemp ของ "วันถัดไป"
data['NextMeanTemp'] = data['MeanTemp'].shift(-1)

# ลบแถวที่ target เป็น NaN (เพราะวันสุดท้ายไม่มีวันถัดไป)
data = data.dropna()

# แยก X, y
X = data[['MaxTemp', 'MinTemp', 'Precip']]
y = data['NextMeanTemp']

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print(f"\n🎯 แบ่งข้อมูล:")
print(f"   - ข้อมูลเทรน: {len(X_train)} วัน")
print(f"   - ข้อมูลทดสอบ: {len(X_test)} วัน")

# เทรนโมเดล
print(f"\n🤖 กำลังฝึกโมเดล AI...")
model = LinearRegression()
model.fit(X_train, y_train)

# ประเมินโมเดล
y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

print(f"\n📊 ผลการประเมิน AI:")
print(f"   🎯 MAE: {mae:.2f}°C") # โดยเฉลี่ยแล้ว AI คาดเคลื่อนจากค่า MeanTemp จริง ๆ ประมาณกี่องศา
print(f"   📏 RMSE: {rmse:.2f}°C") # โดยทั่วไป AI ทำนายคลาดเคลื่อนประมาณกี่องศาเซลเซียส
print(f"   📈 R² Score: {r2:.4f} ({r2*100:.1f}%)")  # ความแม่นยำกีเปอร์เซ็นต์

# MAE (Mean Absolute Error) ค่าความผิดพลาดเฉลี่ยแบบ ไม่สนใจบวกหรือลบ (ดูแต่ขนาดว่าคลาดเคลื่อนกี่องศา)
# RMSE (Root Mean Squared Error) ค่ารากที่สองของความคลาดเคลื่อนกำลังสอง → เน้น “การลงโทษข้อผิดพลาดใหญ่ ๆ” มากกว่าข้อผิดพลาดเล็ก ๆ
# R² Score (Coefficient of Determination) บอกว่าโมเดลอธิบายความแปรผันของข้อมูลจริงได้ดีแค่ไหน

# บันทึกโมเดล
joblib.dump(model, "weather_model.pkl")
print("\n💾 โมเดลถูกบันทึกเป็นไฟล์: weather_model.pkl")

# แสดงตัวอย่างผลทำนาย 10 วันจากชุด test
print("\n🔎 ตัวอย่างผลทำนาย 10 วันจากชุด Test:")
sample_results = pd.DataFrame({
    "MaxTemp |": X_test["MaxTemp"].values,
    "MinTemp |": X_test["MinTemp"].values,
    "Precip |": X_test["Precip"].values,
    "ค่าจริง |": y_test.values, # MeanTemp
    "ค่าทำนาย": y_pred
})

# สุ่มเลือก 10 แถวจาก test set
print(sample_results.sample(10, random_state=42).to_string(index=False,
    formatters={
        "MaxTemp |": "{:.2f}°C".format,
        "MinTemp |": "{:.2f}°C".format,
        "ค่าจริง |": "{:.2f}°C".format,
        "ค่าทำนาย": "{:.2f}°C".format
    }))