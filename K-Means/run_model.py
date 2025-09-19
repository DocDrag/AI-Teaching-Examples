import pandas as pd
import joblib
import pickle

# โหลดโมเดล, scaler และ mapping
kmeans = joblib.load('customer_kmeans.pkl')
scaler = joblib.load('customer_scaler.pkl')

with open("cluster_names.pkl", "rb") as f:
    cluster_names = pickle.load(f)

# โหลดข้อมูลลูกค้าเดิม
try:
    df_all = joblib.load('customer_data.pkl')
except FileNotFoundError:
    # ถ้าหาไฟล์ไม่เจอ
    df_all = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary', 'CustomerID', 'Cluster'])

# ฟังก์ชันทำนายกลุ่ม
def predict_customer_group(customer_data):
    df_new = pd.DataFrame([customer_data], columns=['Recency', 'Frequency', 'Monetary', 'CustomerID'])
    X_scaled = scaler.transform(df_new[['Recency', 'Frequency', 'Monetary']])
    cluster_id = kmeans.predict(X_scaled)[0]
    group_name = cluster_names.get(cluster_id, "Unknown")
    df_new['Cluster'] = cluster_id
    return df_new, cluster_id, group_name

# ฟังก์ชันแนะนำกลยุทธ์
def get_ad_strategy(group_name):
    if group_name == "💎 ลูกค้า VIP":
        return {"โฆษณา": "🌟 สินค้าพรีเมียมล่าสุด - เฉพาะคุณเท่านั้น!", "ข้อเสนอ": "ส่วนลด 15% + ส่งฟรี"}
    elif group_name == "💰 ลูกค้าประหยัด":
        return {"โฆษณา": "🔥 Flash Sale สุดคุ้ม! ลดสูงสุด 50%", "ข้อเสนอ": "ซื้อ 2 แถม 1 + คูปอง 100 บาท"}
    else:  # 😴 ลูกค้าไม่แน่นอน
        return {"โฆษณา": "💔 เราคิดถึงคุณ!", "ข้อเสนอ": "คูปองส่วนลด 30% (7 วัน)"}

# ------------------ MAIN ------------------
if __name__ == "__main__":
    # ✏️ ใส่ค่าลูกค้าที่นี่
    Recency = 30 # วัน
    Frequency = 12 # ครั้ง/ปี
    Monetary = 15000 # บาท/ปี
    CustomerID = "CUST001"
    customer_data = [Recency, Frequency, Monetary, CustomerID]

    # ทำนายกลุ่ม
    df_new, cluster_id, group_name = predict_customer_group(customer_data)
    strategy = get_ad_strategy(group_name)

    # รวมข้อมูลเก่าแล้วบันทึก
    updated_df = pd.concat([df_all, df_new], ignore_index=True)
    joblib.dump(updated_df, 'customer_data.pkl')

    # แสดงผล
    print("👤 ลูกค้า:", customer_data[3])
    print("🎯 กลุ่ม:", group_name)
    print("📢 โฆษณา:", strategy['โฆษณา'])
    print("🎁 ข้อเสนอ:", strategy['ข้อเสนอ'])
