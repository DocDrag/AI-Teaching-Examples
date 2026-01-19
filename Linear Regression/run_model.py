import pandas as pd
import joblib

# โหลดโมเดลที่บันทึกไว้
print("กำลังโหลดโมเดล...")
model = joblib.load("weather_model.pkl")
print("โหลดโมเดลสำเร็จ!")

# ทดสอบการทำนายกับข้อมูลใหม่
test_cases = [
    {"name": "วันแดดร้อน", "MaxTemp": 35, "MinTemp": 25, "Precip": 0},
    {"name": "วันฝนตก", "MaxTemp": 28, "MinTemp": 22, "Precip": 15},
    {"name": "วันหนาวเย็น", "MaxTemp": 20, "MinTemp": 10, "Precip": 0},
    {"name": "วันปกติ", "MaxTemp": 30, "MinTemp": 24, "Precip": 2}
]

print("\nทดสอบการทำนายของ AI:")
print("=" * 50)

for test_case in test_cases:
    # เตรียมข้อมูล DataFrame สำหรับทำนาย
    test_data = pd.DataFrame([[test_case["MaxTemp"], test_case["MinTemp"], test_case["Precip"]]],
                             columns=['MaxTemp', 'MinTemp', 'Precip'])

    predicted_temp = model.predict(test_data)[0]

    print(f"\n{test_case['name']}:")
    print(
        f"   ข้อมูลนำเข้า: MaxTemp={test_case['MaxTemp']}°C, MinTemp={test_case['MinTemp']}°C, Precip={test_case['Precip']} mm")
    print(f"   ผลการทำนาย (MeanTemp ของวันถัดไป): {predicted_temp:.2f}°C")
