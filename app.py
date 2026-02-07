# =============================================================
# app.py - Face Recognition Dashboard
# ระบบจำแนกใบหน้าด้วย PCA + SVM แสดงผลผ่าน Streamlit
# =============================================================

# --- Import Libraries ---
import streamlit as st          # สร้างหน้าเว็บ Dashboard
import numpy as np              # จัดการ array/ตัวเลข
import matplotlib.pyplot as plt # สร้างกราฟ
import seaborn as sns           # สร้าง heatmap สวยๆ
import pandas as pd             # จัดการข้อมูลตาราง

# --- Import ML Libraries ---
from sklearn.datasets import fetch_lfw_people          # โหลด dataset ใบหน้า
from sklearn.model_selection import train_test_split   # แบ่ง train/test
from sklearn.model_selection import GridSearchCV       # หาค่า parameter ดีสุด
from sklearn.decomposition import PCA                  # ลดมิติข้อมูล
from sklearn.svm import SVC                            # โมเดล SVM
from sklearn.pipeline import make_pipeline             # รวม PCA+SVM เป็น pipeline
from sklearn.metrics import classification_report      # รายงานผล
from sklearn.metrics import confusion_matrix           # ตาราง confusion
from sklearn.metrics import accuracy_score             # คำนวณ accuracy

# =============================================================
# ตั้งค่าหน้าเว็บ
# =============================================================
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="wide"  # ใช้หน้าจอเต็ม
)

# --- แสดงหัวข้อ ---
st.title("🧠 Face Recognition System")
st.markdown("**Using PCA + SVM with GridSearchCV Optimization**")
st.markdown("---")

# =============================================================
# Sidebar - ตั้งค่าโมเดล
# =============================================================
st.sidebar.header("⚙️ Model Settings")

# ให้ user เลือกค่าต่างๆ
n_components = st.sidebar.slider("PCA Components", 50, 300, 150, step=10)
# จำนวน component ของ PCA (ลดจาก ~3000 pixel เหลือเท่าไหร่)

min_faces = st.sidebar.slider("Min Faces per Person", 40, 100, 60, step=10)
# เอาเฉพาะคนที่มีรูปอย่างน้อยกี่รูป

test_size = st.sidebar.slider("Test Size (%)", 10, 40, 25, step=5) / 100
# แบ่งข้อมูลทดสอบกี่ %

# =============================================================
# ฟังก์ชันโหลดข้อมูลและเทรนโมเดล
# =============================================================
@st.cache_data  # cache ไว้ไม่ต้องโหลดใหม่ทุกครั้ง
def load_and_train(n_components, min_faces, test_size):
    """
    โหลด dataset, เทรนโมเดล, return ผลลัพธ์
    """
    
    # --- 1. โหลดข้อมูล LFW ---
    faces = fetch_lfw_people(min_faces_per_person=min_faces)
    n_samples, h, w = faces.images.shape  # จำนวนรูป, ความสูง, ความกว้าง
    X = faces.data          # ข้อมูล pixel (flatten เป็น 1 มิติแล้ว)
    y = faces.target        # label (ใครคือใคร)
    target_names = faces.target_names  # ชื่อคน
    
    # --- 2. แบ่ง Train / Test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # --- 3. สร้าง Pipeline: PCA -> SVM ---
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    # whiten=True ทำให้แต่ละ component มี variance เท่ากัน
    
    svc = SVC(kernel='rbf', class_weight='balanced')
    # kernel='rbf' เหมาะกับข้อมูลซับซ้อน
    # class_weight='balanced' ให้ความสำคัญทุก class เท่ากัน
    
    model = make_pipeline(pca, svc)  # รวมเป็น pipeline เดียว
    
    # --- 4. GridSearch หาค่า C และ gamma ดีสุด ---
    param_grid = {
        'svc__C': [1, 5, 10],           # ค่าความเข้มงวด
        'svc__gamma': [0.001, 0.005, 0.01]  # ขอบเขตการตัดสินใจ
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    # cv=3: ใช้ 3-fold cross validation
    # n_jobs=-1: ใช้ทุก CPU core
    
    grid.fit(X_train, y_train)  # เทรนโมเดล
    
    # --- 5. ทำนายผล ---
    y_pred = grid.predict(X_test)
    
    # --- Return ผลลัพธ์ทั้งหมด ---
    return {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_names': target_names,
        'best_params': grid.best_params_,   # ค่า C, gamma ที่ดีสุด
        'best_score': grid.best_score_,     # คะแนน CV ที่ดีสุด
        'h': h,
        'w': w,
        'n_samples': n_samples,
        'n_classes': len(target_names)
    }

# =============================================================
# ปุ่มกด Train Model
# =============================================================
if st.sidebar.button("🚀 Train Model", type="primary"):
    with st.spinner("Training model... This may take 1-2 minutes."):
        results = load_and_train(n_components, min_faces, test_size)
        st.session_state['results'] = results  # เก็บผลไว้ใน session
        st.success("Model trained successfully!")

# =============================================================
# แสดงผลลัพธ์ (ถ้าเทรนแล้ว)
# =============================================================
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # --- แสดง Metrics ---
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = accuracy_score(results['y_test'], results['y_pred'])
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Best C", results['best_params']['svc__C'])
    col3.metric("Best Gamma", results['best_params']['svc__gamma'])
    col4.metric("CV Score", f"{results['best_score']:.1%}")
    
    st.markdown("---")
    
    # --- แบ่ง 2 คอลัมน์สำหรับกราฟ ---
    col_left, col_right = st.columns(2)
    
    # --- Confusion Matrix (ซ้าย) ---
    with col_left:
        st.subheader("🔥 Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[n.split()[-1] for n in results['target_names']],
                    yticklabels=[n.split()[-1] for n in results['target_names']], ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig1)
    
    # --- Pie Chart (ขวา) ---
    with col_right:
        st.subheader("🥧 Prediction Distribution")
        pred_counts = pd.Series(results['y_pred']).value_counts().sort_index()
        labels = [results['target_names'][i].split()[-1] for i in pred_counts.index]
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax2.pie(pred_counts.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Predicted Classes')
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # --- Classification Report (ตาราง) ---
    st.subheader("📋 Classification Report")
    report = classification_report(results['y_test'], results['y_pred'], 
                                   target_names=results['target_names'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # --- แสดงตัวอย่างการทำนาย ---
    st.subheader("🖼️ Sample Predictions")
    n_samples_show = min(10, len(results['X_test']))
    cols = st.columns(5)
    
    for i in range(n_samples_show):
        with cols[i % 5]:
            # แปลง pixel กลับเป็นรูป
            img = results['X_test'][i].reshape(results['h'], results['w'])
            pred_name = results['target_names'][results['y_pred'][i]].split()[-1]
            true_name = results['target_names'][results['y_test'][i]].split()[-1]
            correct = results['y_pred'][i] == results['y_test'][i]
            
            st.image(img, caption=f"Pred: {pred_name}", use_container_width=True)
            if correct:
                st.success(f"✓ {true_name}")  # ทายถูก
            else:
                st.error(f"✗ Actual: {true_name}")  # ทายผิด

# =============================================================
# ถ้ายังไม่ได้เทรน แสดงข้อมูลเบื้องต้น
# =============================================================
else:
    st.info("👈 Click **Train Model** in the sidebar to start!")
    
    st.subheader("📚 About the Dataset")
    st.markdown("""
    This project uses the **LFW (Labeled Faces in the Wild)** dataset:
    - Contains face images of famous people
    - Images are 62x47 pixels (grayscale)
    - Used for face recognition benchmarking
    
    **Techniques Used:**
    - **PCA**: Reduces dimensions from ~3000 pixels to 150 components
    - **SVM (RBF)**: Classifies faces using support vectors
    - **GridSearchCV**: Finds optimal hyperparameters (C, gamma)
    """)

# =============================================================
# Footer
# =============================================================
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Face Recognition System")
