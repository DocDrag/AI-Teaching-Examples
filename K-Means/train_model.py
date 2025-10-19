import pandas as pd
from sklearn.cluster import KMeans
import joblib
import pickle

# อ่านไฟล์ที่ถูกสเกลแล้ว
df = pd.read_csv('customer_scaled_data.csv')

# เตรียมข้อมูล
X = df[['Recency', 'Frequency', 'Monetary']].values

# ฝึกโมเดล K-Means
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# วิเคราะห์ผล
print("\n=== การวิเคราะห์แต่ละกลุ่มลูกค้า ===")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n📊 กลุ่มที่ {cluster}:")
    print(f"   จำนวนลูกค้า: {len(cluster_data)} คน")
    print(f"   เฉลี่ย Recency: {cluster_data['Recency'].mean():.1f}")
    print(f"   เฉลี่ย Frequency: {cluster_data['Frequency'].mean():.1f}")
    print(f"   เฉลี่ย Monetary: {cluster_data['Monetary'].mean():.1f}")

# ตั้งชื่อกลุ่ม
cluster_names = {}
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    avg_r = cluster_data['Recency'].mean()
    avg_f = cluster_data['Frequency'].mean()
    avg_m = cluster_data['Monetary'].mean()

    if avg_m > 0.8 and avg_f > 0.7:
        cluster_names[cluster] = "💎 ลูกค้า VIP"
    elif avg_f > 0.5 and avg_m < 0.4:
        cluster_names[cluster] = "💰 ลูกค้าประหยัด"
    else:
        cluster_names[cluster] = "😴 ลูกค้าไม่แน่นอน"

# บันทึกโมเดลและข้อมูล
joblib.dump(kmeans, 'customer_kmeans.pkl')
joblib.dump(df, 'customer_data.pkl')

with open("cluster_names.pkl", "wb") as f:
    pickle.dump(cluster_names, f)

print("\n✅ บันทึกโมเดลและข้อมูลเรียบร้อยแล้ว")
print("   → customer_kmeans.pkl")
print("   → customer_data.pkl")
print("   → cluster_names.pkl")
