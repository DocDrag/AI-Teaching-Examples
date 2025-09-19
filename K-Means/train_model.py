import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# อ่านไฟล์ customer_clean_data.csv
data = pd.read_csv('customer_clean_data.csv')
df = pd.DataFrame(data, columns=['Recency', 'Frequency', 'Monetary', 'CustomerID'])

print("=== ข้อมูลลูกค้าตัวอย่าง ===")
print(df.head(10))
print(f"\nจำนวนลูกค้าทั้งหมด: {len(df)} คน")
print("\n=== สถิติพื้นฐาน ===")
print(df[['Recency', 'Frequency', 'Monetary']].describe())

# เตรียมข้อมูล
X = df[['Recency', 'Frequency', 'Monetary']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# หา k ที่เหมาะสมด้วย Elbow Method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

# เลือก k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n=== การวิเคราะห์แต่ละกลุ่มลูกค้า ===")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n📊 กลุ่มที่ {cluster}:")
    print(f"   จำนวนลูกค้า: {len(cluster_data)} คน")
    print(f"   ซื้อครั้งสุดท้ายเมื่อ: {cluster_data['Recency'].mean():.1f} วัน")
    print(f"   ความถี่การซื้อ: {cluster_data['Frequency'].mean():.1f} ครั้ง/ปี")
    print(f"   ใช้จ่ายเฉลี่ย: {cluster_data['Monetary'].mean():.0f} บาท/ปี")

# สร้าง mapping Cluster → ชื่อกลุ่ม
cluster_names = {}
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    avg_r = cluster_data['Recency'].mean()
    avg_f = cluster_data['Frequency'].mean()
    avg_m = cluster_data['Monetary'].mean()

    if avg_m > 8000 and avg_f > 15:
        cluster_names[cluster] = "💎 ลูกค้า VIP"
    elif avg_f > 8 and avg_m < 4000:
        cluster_names[cluster] = "💰 ลูกค้าประหยัด"
    else:
        cluster_names[cluster] = "😴 ลูกค้าไม่แน่นอน"

print("\n📌 Mapping Cluster → ชื่อกลุ่ม:")
for k, v in cluster_names.items():
    print(f"   Cluster {k}: {v}")

# บันทึกโมเดลและ scaler
joblib.dump(kmeans, 'customer_kmeans.pkl')
joblib.dump(scaler, 'customer_scaler.pkl')

# บันทึกข้อมูลลูกค้าเริ่มต้น
joblib.dump(df, 'customer_data.pkl')

# บันทึก mapping Cluster → ชื่อกลุ่ม
with open("cluster_names.pkl", "wb") as f:
    pickle.dump(cluster_names, f)

print("\n✅ บันทึกโมเดล, scaler, ข้อมูลลูกค้า และ mapping เรียบร้อยแล้ว")
print("   → customer_kmeans.pkl")
print("   → customer_scaler.pkl")
print("   → customer_data.pkl")
print("   → cluster_names.pkl")
