import pandas as pd
import joblib
import pickle
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import messagebox

# โหลดโมเดล, scaler และ mapping
kmeans = joblib.load('customer_kmeans.pkl')
scaler = joblib.load('customer_scaler.pkl')

with open("cluster_names.pkl", "rb") as f:
    cluster_names = pickle.load(f)

# โหลดข้อมูลลูกค้าเดิม
try:
    df_all = joblib.load('customer_data.pkl')
except FileNotFoundError:
    df_all = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary', 'CustomerID', 'Cluster'])

# ฟังก์ชันทำนายกลุ่ม
def predict_customer_group(customer_data):
    df_new = pd.DataFrame([customer_data], columns=['Recency', 'Frequency', 'Monetary', 'CustomerID'])
    X_scaled = scaler.transform(df_new[['Recency', 'Frequency', 'Monetary']])
    cluster_id = kmeans.predict(X_scaled)[0]
    group_name = cluster_names.get(cluster_id, "Unknown")
    df_new['Cluster'] = cluster_id
    return df_new, cluster_id, group_name

def get_ad_strategy(group_name):
    if group_name == "💎 ลูกค้า VIP":
        return {"โฆษณา": "🌟 สินค้าพรีเมียมล่าสุด - เฉพาะคุณเท่านั้น!", "ข้อเสนอ": "ส่วนลด 15% + ส่งฟรี"}
    elif group_name == "💰 ลูกค้าประหยัด":
        return {"โฆษณา": "🔥 Flash Sale สุดคุ้ม! ลดสูงสุด 50%", "ข้อเสนอ": "ซื้อ 2 แถม 1 + คูปอง 100 บาท"}
    else:  # 😴 ลูกค้าไม่แน่นอน
        return {"โฆษณา": "💔 เราคิดถึงคุณ!", "ข้อเสนอ": "คูปองส่วนลด 30% (7 วัน)"}

def save_and_predict():
    try:
        recency = float(entry_recency.get())
        frequency = float(entry_frequency.get())
        monetary = float(entry_monetary.get())
        customer_id = entry_customer_id.get().strip()

        if not customer_id:
            messagebox.showerror("ข้อผิดพลาด", "กรุณากรอก CustomerID")
            return

        customer_data = [recency, frequency, monetary, customer_id]
        df_new, cluster_id, group_name = predict_customer_group(customer_data)
        strategy = get_ad_strategy(group_name)

        # รวมข้อมูลเก่า
        updated_df = pd.concat([df_all, df_new], ignore_index=True)
        joblib.dump(updated_df, 'customer_data.pkl')

        # แสดงผล
        result_text.set(f"👤 ลูกค้า: {customer_id}\n🎯 กลุ่ม: {group_name}\n📢 โฆษณา: {strategy['โฆษณา']}\n🎁 ข้อเสนอ: {strategy['ข้อเสนอ']}")

    except ValueError:
        messagebox.showerror("ข้อผิดพลาด", "กรุณากรอกตัวเลขใน Recency, Frequency, Monetary ให้ถูกต้อง")

def clear():
    entry_recency.delete(0, 'end')
    entry_frequency.delete(0, 'end')
    entry_monetary.delete(0, 'end')
    entry_customer_id.delete(0, 'end')
    result_text.set("")

# ---------------- UI ----------------
root = tb.Window(themename="cosmo")
root.title("Customer Segmentation")
root.geometry("600x500")

title_label = tb.Label(root, text="📊 Customer Segmentation", font=("Tahoma", 18, "bold"))
title_label.pack(pady=15)

form_frame = tb.Frame(root)
form_frame.pack(pady=10)

fields = [
    ("Customer ID", "entry_customer_id"),
    ("ซื้อครั้งล่าสุด (กี่วันก่อน)", "entry_recency"),
    ("ซื้อบ่อยแค่ไหน (ครั้ง/ปี)", "entry_frequency"),
    ("ใช้เงินเท่าไหร่ (บาท/ปี)", "entry_monetary")
]

entry_recency = None
entry_frequency = None
entry_monetary = None
entry_customer_id = None

for i, (label, var_name) in enumerate(fields):
    tb.Label(form_frame, text=label, font=("Tahoma", 12)).grid(row=i, column=0, sticky="w", padx=10, pady=5)
    entry = tb.Entry(form_frame, font=("Tahoma", 12), width=25)
    entry.grid(row=i, column=1, padx=10, pady=5)
    globals()[var_name] = entry

btn_frame = tb.Frame(root)
btn_frame.pack(pady=15)

tb.Button(btn_frame, text="บันทึกและทำนาย", bootstyle=SUCCESS, command=save_and_predict).grid(row=0, column=0, padx=10)
tb.Button(btn_frame, text="ล้างค่า", bootstyle=WARNING, command=clear).grid(row=0, column=1, padx=10)

result_text = tb.StringVar()
tb.Label(root, textvariable=result_text, font=("Tahoma", 12), wraplength=500, justify="left", foreground="#0066CC").pack(pady=15)

root.mainloop()
