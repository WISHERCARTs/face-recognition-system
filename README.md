# Face Recognition System 🧠

โปรเจกต์ระบบจำแนกใบหน้าโดยใช้ Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wishercarts-face-recognition-system-app-vti7zr.streamlit.app/)

👉 **[Live Demo](https://wishercarts-face-recognition-system-app-vti7zr.streamlit.app/)**

## ทำอะไร?

ระบบนี้ใช้ **PCA** ลดขนาดข้อมูลรูปภาพ แล้วใช้ **SVM** จำแนกว่าเป็นใบหน้าของใคร

## Dataset

ใช้ชุดข้อมูล **LFW (Labeled Faces in the Wild)** จาก sklearn

- รูปใบหน้าคนดัง
- ขนาด 62x47 pixels

## วิธีรัน

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน training script
python Faces.py

# รัน dashboard
streamlit run app.py
```

## ไฟล์ในโปรเจกต์

| ไฟล์               | อธิบาย                     |
| ------------------ | -------------------------- |
| `Faces.py`         | โค้ดหลักสำหรับ train model |
| `app.py`           | Dashboard แสดงผลลัพธ์      |
| `requirements.txt` | รายการ library ที่ใช้      |

## เทคนิคที่ใช้

1. **PCA** - ลด features จาก ~3000 เหลือ 150
2. **SVM (RBF kernel)** - จำแนกใบหน้า
3. **GridSearchCV** - หาค่า parameter ที่ดีที่สุด

## ผลลัพธ์

- Accuracy ประมาณ 85-90%
- แสดง Confusion Matrix และ Pie Chart

## สิ่งที่เรียนรู้

- การใช้ PCA สำหรับ dimensionality reduction
- การ train SVM classifier
- การปรับจูน hyperparameters ด้วย GridSearchCV
- การ visualize ผลลัพธ์ด้วย matplotlib และ seaborn

---

Made by ["Wish Nakthong"]
