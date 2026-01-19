import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# โหลดข้อมูล
print("กำลังโหลดข้อมูลสภาพอากาศที่ทำความสะอาดแล้ว...")
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

print(f"\nแบ่งข้อมูล:")
print(f"   - ข้อมูลเทรน: {len(X_train)} วัน")
print(f"   - ข้อมูลทดสอบ: {len(X_test)} วัน")

# เทรนโมเดล
print(f"\nกำลังฝึกโมเดล AI...")
model = LinearRegression()
model.fit(X_train, y_train)

# ประเมินโมเดล
y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

print(f"\nผลการประเมิน AI:")
print(f"   MAE: {mae:.2f}°C")
print(f"   RMSE: {rmse:.2f}°C")
print(f"   R² Score: {r2:.4f} ({r2*100:.1f}%)")

# บันทึกโมเดล
joblib.dump(model, "weather_model.pkl")
print("\nโมเดลถูกบันทึกเป็นไฟล์: weather_model.pkl")

# แสดงตัวอย่างผลทำนาย 10 วันจากชุด test
print("\nตัวอย่างผลทำนาย 10 วันจากชุด Test:")
sample_results = pd.DataFrame({
    "MaxTemp |": X_test["MaxTemp"].values,
    "MinTemp |": X_test["MinTemp"].values,
    "Precip |": X_test["Precip"].values,
    "ค่าจริง |": y_test.values, # MeanTemp
    "ค่าทำนาย": y_pred
})

# print(sample_results.sample(10))
# สุ่มเลือก 10 แถวจาก test set
print(sample_results.sample(10, random_state=42).to_string(index=False,
    formatters={
        "MaxTemp |": "{:.2f}°C".format,
        "MinTemp |": "{:.2f}°C".format,
        "ค่าจริง |": "{:.2f}°C".format,
        "ค่าทำนาย": "{:.2f}°C".format
    }))