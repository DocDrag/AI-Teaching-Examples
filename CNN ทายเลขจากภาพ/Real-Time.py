import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import os


class RealTimeDigitRecognizer:
    def __init__(self):
        # โหลดโมเดล
        self.model_path = 'cnn_mnist_augmented_model.keras'  # CNN model

        self.model = None

        # ลองโหลดโมเดลตามลำดับ
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except:
            print("Error: Could not load model. Please make sure 'cnn_mnist_augmented_model.keras' exists.")

        if self.model is None:
            print("❌ Error: Could not load any model!")
            print("📁 Please make sure one of these files exists:")
            for path in self.model_path:
                print(f"   - {path}")
            return

        # สร้าง GUI
        self.root = tk.Tk()
        self.root.title(f"Real-time Digit Recognition - CNN Model")
        self.root.geometry("900x650")

        # ตัวแปรสำหรับการวาด
        self.canvas_size = 280
        self.brush_size = 30  # ขนาดพู่กัน

        # สร้าง Canvas สำหรับวาด
        self.drawing_canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        self.drawing_canvas.pack(side=tk.LEFT, padx=20, pady=20)

        # สร้าง PIL Image สำหรับเก็บข้อมูลการวาด
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Bind events สำหรับการวาด
        self.drawing_canvas.bind("<Button-1>", self.start_drawing)
        self.drawing_canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.drawing_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.is_drawing = False
        self.last_x = None
        self.last_y = None

        # สร้างปุ่ม
        self.create_button_panel()

        # สร้างส่วนแสดงผล
        self.create_result_panel()

        # เริ่มการทำนายแบบเรียลไทม์
        self.prediction_active = True
        self.prediction_thread = threading.Thread(target=self.continuous_prediction, daemon=True)
        self.prediction_thread.start()

    def create_result_panel(self):
        """สร้างพาเนลแสดงผลการทำนาย"""
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.LEFT, padx=20, pady=0, fill=tk.BOTH, expand=True)

        # หัวข้อ
        title_label = tk.Label(
            right_frame,
            text=f"ผลการทำนาย",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=5)

        # แสดงตัวเลขที่ทำนายได้
        self.prediction_label = tk.Label(
            right_frame,
            text="วาดตัวเลข 0-9",
            font=("Arial", 28, "bold"),
            fg="blue"
        )
        self.prediction_label.pack(pady=10)

        # แสดงความมั่นใจ
        self.confidence_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 14),
            fg="green"
        )
        self.confidence_label.pack(pady=5)

        # กราฟแสดงความน่าจะเป็นของแต่ละตัวเลข
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas_plot.get_tk_widget().pack(pady=10)

        # แสดงข้อมูลโมเดล
        model_info = tk.Label(
            right_frame,
            text=f"Input shape: {self.model.input_shape}",
            font=("Arial", 10),
            fg="gray"
        )
        model_info.pack(pady=5)

    def create_button_panel(self):
        """สร้างพาเนลแสดงปุ่มควบคุม"""
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

        # ปุ่มล้างหน้าจอ
        clear_button = tk.Button(
            button_frame,
            text="🗑️ ล้างหน้าจอ",
            command=self.clear_canvas,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=5
        )
        clear_button.pack(side=tk.LEFT, padx=10)

        # ปุ่มบันทึกภาพ
        save_button = tk.Button(
            button_frame,
            text="💾 บันทึกภาพ",
            command=self.save_image,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=5
        )
        save_button.pack(side=tk.LEFT, padx=10)

        # ปุ่มแสดงภาพที่โมเดลเห็น
        preview_button = tk.Button(
            button_frame,
            text="👁️ ดูภาพที่โมเดลเห็น",
            command=self.show_model_view,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=5
        )
        preview_button.pack(side=tk.LEFT, padx=10)

    def start_drawing(self, event):
        """เริ่มการวาด"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_on_canvas(self, event):
        """วาดบน Canvas"""
        if self.is_drawing and self.last_x and self.last_y:
            # วาดบน Tkinter Canvas
            self.drawing_canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill='black', width=self.brush_size, capstyle=tk.ROUND
            )

            # วาดบน PIL Image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill='black', width=self.brush_size
            )

            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        """หยุดการวาด"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """ล้างหน้าจอ"""
        self.drawing_canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # รีเซ็ตการแสดงผล
        self.prediction_label.config(text="วาดตัวเลข 0-9", fg="gray")
        self.confidence_label.config(text="")
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Draw numbers to see the result.', transform=self.ax.transAxes,
                     ha='center', va='center', fontsize=14, color='gray')
        self.canvas_plot.draw()

    def save_image(self):
        """บันทึกภาพ"""
        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            folder_name = "เลขที่พลาด"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f"📁 Created folder: {folder_name}")

            timestamp = int(time.time())

            # ได้ผลการทำนายปัจจุบัน
            predicted_class, probabilities = self.predict_digit()
            if predicted_class is not None:
                confidence = probabilities[predicted_class]
                # เพิ่มข้อมูลการทำนายในชื่อไฟล์
                base_name = f"prediction_{predicted_class}_conf_{confidence:.0%}_{timestamp}"
            else:
                base_name = f"empty_drawing_{timestamp}"

            # บันทึกภาพที่โมเดลเห็น
            processed_filename = os.path.join(folder_name, f"{base_name}_processed.png")
            _, centered_img = self.preprocess_image()
            processed_img = Image.fromarray((centered_img * 255).astype(np.uint8), 'L')
            processed_img.save(processed_filename)

            print(f"✅ Processed image saved: {processed_filename}")

            # แสดง popup แจ้งเตือน
            self.show_save_notification(folder_name, predicted_class, confidence if predicted_class else 0)

        except Exception as e:
            print(f"❌ Error saving image: {e}")
            # แสดง error popup
            error_window = tk.Toplevel(self.root)
            error_window.title("เกิดข้อผิดพลาด")
            error_window.geometry("300x100")
            tk.Label(error_window, text=f"ไม่สามารถบันทึกไฟล์ได้\n{str(e)}",
                     font=("Arial", 10), fg="red").pack(expand=True)

    def show_save_notification(self, folder_name, predicted_class, confidence):
        """แสดงการแจ้งเตือนการบันทึก"""
        notification = tk.Toplevel(self.root)
        notification.title("บันทึกสำเร็จ")
        notification.geometry("350x120")
        notification.resizable(False, False)

        # ทำให้อยู่ตรงกลาง
        notification.transient(self.root)
        notification.grab_set()

        if predicted_class is not None:
            message = f"✅ บันทึกเรียบร้อย!\n📁 โฟลเดอร์: {folder_name}\n🎯 ทำนาย: {predicted_class} ({confidence:.0%})"
        else:
            message = f"✅ บันทึกเรียบร้อย!\n📁 โฟลเดอร์: {folder_name}\n📝 ภาพว่าง"

        tk.Label(notification, text=message, font=("Arial", 11), justify="left").pack(pady=20)

        tk.Button(notification, text="ตกลง", command=notification.destroy,
                 bg="#3498db", fg="white", font=("Arial", 10), padx=20).pack()

        # ปิดอัตโนมัติหลัง 3 วินาทีา
        notification.after(3000, notification.destroy)

    def show_model_view(self):
        """แสดงภาพที่โมเดลเห็น (หลังจัดให้อยู่กลาง)"""
        _, centered_img = self.preprocess_image()  # ใช้ภาพที่จัดกลางแล้ว

        # สร้างหน้าต่างใหม่
        preview_window = tk.Toplevel(self.root)
        preview_window.title("ภาพที่โมเดลเห็น (28x28)")
        preview_window.geometry("400x400")

        # แสดงภาพ
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(centered_img, cmap='gray')
        ax.set_title('model sees (28x28)', fontsize=14)
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, preview_window)
        canvas.get_tk_widget().pack(expand=True, fill='both')
        canvas.draw()

    def preprocess_for_display(self):
        """เตรียมภาพสำหรับแสดงผล"""
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        # กลับสี สำหรับ display
        img_array = 255 - img_array
        img_array = img_array / 255.0
        return img_array

    def center_image(self, img_array):
        """จัดภาพให้อยู่กลางตามจุดศูนย์กลางของมวล"""
        from scipy.ndimage import shift
        from numpy import isnan

        # ตรวจว่าภาพมีเนื้อหาหรือไม่
        if np.sum(img_array) == 0:
            return img_array  # ภาพว่างเปล่า ไม่ต้องจัดกลาง

        cy, cx = center_of_mass(img_array)

        if isnan(cx) or isnan(cy):
            return img_array  # ป้องกัน NaN

        shift_x = int(np.round(14 - cx))
        shift_y = int(np.round(14 - cy))

        shifted = shift(img_array, shift=(shift_y, shift_x), mode='constant', cval=0.0)
        return shifted

    def preprocess_image(self):
        """เตรียมภาพสําหรับการทํานาย"""
        # Resize เป็น 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)

        # กลับสี (ดํา->ขาว, ขาว->ดํา)
        img_array = 255 - img_array

        # Normalize
        img_array = img_array.astype('float32') / 255.0

        # กรองภาพ noise ก่อนจัดกลาง (optional)
        if np.max(img_array) < 0.1:
            return np.zeros((1, 28, 28, 1)), img_array  # ป้องกันค่าผิดปกติ

        # จัดภาพให้อยู่กลาง
        img_array = self.center_image(img_array)

        # เตรียม input
        img_for_prediction = img_array.reshape(1, 28, 28, 1)

        return img_for_prediction, img_array

    def predict_digit(self):
        """ทำนายตัวเลข"""
        try:
            img_for_prediction, processed_img = self.preprocess_image()

            # ตรวจสอบว่ามีการวาดหรือไม่
            if np.sum(processed_img) < 0.05:  # ถ้าภาพเกือบว่างเปล่า
                return None, None

            # ทำนาย
            prediction_probabilities = self.model.predict(img_for_prediction, verbose=0)
            predicted_class = np.argmax(prediction_probabilities)

            return predicted_class, prediction_probabilities[0]

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None

    def update_display(self, predicted_class, probabilities):
        """อัพเดทการแสดงผล"""
        if predicted_class is not None:
            # อัพเดทป้ายแสดงผล
            self.prediction_label.config(
                text=f"🎯 {predicted_class}",
                fg="blue"
            )

            confidence = probabilities[predicted_class]
            confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
            self.confidence_label.config(
                text=f"ความมั่นใจ: {confidence:.1%}",
                fg=confidence_color
            )

            # อัพเดทกราฟ
            self.ax.clear()
            colors = ['#e74c3c' if i == predicted_class else '#3498db' for i in range(10)]
            bars = self.ax.bar(range(10), probabilities, color=colors, alpha=0.8)

            self.ax.set_xlabel('number', fontsize=12)
            self.ax.set_ylabel('Probability', fontsize=12)
            self.ax.set_title(f'Prediction: {predicted_class} ({confidence:.1%})', fontsize=14, fontweight='bold')
            self.ax.set_xticks(range(10))
            self.ax.set_ylim(0, 1)
            self.ax.grid(True, alpha=0.3)

            # เพิ่มค่าบนแท่ง
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                if prob > 0.03:  # แสดงเฉพาะค่าที่มากกว่า 3%
                    self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                 f'{prob:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            self.canvas_plot.draw()
        else:
            self.prediction_label.config(text="วาดตัวเลข 0-9", fg="gray")
            self.confidence_label.config(text="")

    def continuous_prediction(self):
        """ทำนายแบบต่อเนื่อง"""
        while self.prediction_active:
            try:
                predicted_class, probabilities = self.predict_digit()

                # อัพเดท UI ใน main thread
                self.root.after(0, lambda: self.update_display(predicted_class, probabilities))

                time.sleep(0.3)  # เร็วขึ้นเล็กน้อย

            except Exception as e:
                print(f"❌ Error in continuous prediction: {e}")
                time.sleep(1)

    def run(self):
        """เริ่มแอปพลิเคชัน"""
        if self.model is None:
            return

        try:
            instructions = f"""
🎯 Real-time Digit Recognition CNN
================================================
📝 คำแนะนำการใช้งาน:
   1. วาดตัวเลข 0-9 บนพื้นที่สีขาวทางซ้าย
   2. ระบบจะทำนายแบบเรียลไทม์
   3. ดูผลลัพธ์และกราฟทางขวา  
   4. ใช้ปุ่มควบคุมด้านล่าง

⚙️  Model Info: CNN
📊 Input Shape: {self.model.input_shape}
            """
            print(instructions)

            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()

        except KeyboardInterrupt:
            print("Application stopped by user")
        except Exception as e:
            print(f"❌ Error running application: {e}")

    def on_closing(self):
        """จัดการเมื่อปิดแอปพลิเคชัน"""
        self.prediction_active = False
        plt.close('all')  # ปิด matplotlib figures
        self.root.destroy()


if __name__ == "__main__":
    app = RealTimeDigitRecognizer()
    app.run()