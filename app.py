# app.py - Face Recognition Dashboard with Streamlit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Face Recognition System")
st.markdown("**Using PCA + SVM with GridSearchCV Optimization**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
n_components = st.sidebar.slider("PCA Components", 50, 300, 150, step=10)
min_faces = st.sidebar.slider("Min Faces per Person", 40, 100, 60, step=10)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 25, step=5) / 100

# Cache the data loading and training
@st.cache_data
def load_and_train(n_components, min_faces, test_size):
    # Load data
    faces = fetch_lfw_people(min_faces_per_person=min_faces)
    n_samples, h, w = faces.images.shape
    X = faces.data
    y = faces.target
    target_names = faces.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Create pipeline
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    
    # GridSearch (simplified for speed)
    param_grid = {
        'svc__C': [1, 5, 10],
        'svc__gamma': [0.001, 0.005, 0.01]
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Predict
    y_pred = grid.predict(X_test)
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_names': target_names,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'h': h,
        'w': w,
        'n_samples': n_samples,
        'n_classes': len(target_names)
    }

# Run button
if st.sidebar.button("🚀 Train Model", type="primary"):
    with st.spinner("Training model... This may take 1-2 minutes."):
        results = load_and_train(n_components, min_faces, test_size)
        st.session_state['results'] = results
        st.success("Model trained successfully!")

# Display results
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Metrics row
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = accuracy_score(results['y_test'], results['y_pred'])
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Best C", results['best_params']['svc__C'])
    col3.metric("Best Gamma", results['best_params']['svc__gamma'])
    col4.metric("CV Score", f"{results['best_score']:.1%}")
    
    st.markdown("---")
    
    # Two columns for charts
    col_left, col_right = st.columns(2)
    
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
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(results['y_test'], results['y_pred'], 
                                   target_names=results['target_names'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Sample predictions
    st.subheader("🖼️ Sample Predictions")
    n_samples_show = min(10, len(results['X_test']))
    cols = st.columns(5)
    
    for i in range(n_samples_show):
        with cols[i % 5]:
            img = results['X_test'][i].reshape(results['h'], results['w'])
            pred_name = results['target_names'][results['y_pred'][i]].split()[-1]
            true_name = results['target_names'][results['y_test'][i]].split()[-1]
            correct = results['y_pred'][i] == results['y_test'][i]
            
            st.image(img, caption=f"Pred: {pred_name}", use_container_width=True)
            if correct:
                st.success(f"✓ {true_name}")
            else:
                st.error(f"✗ Actual: {true_name}")

else:
    st.info("👈 Click **Train Model** in the sidebar to start!")
    
    # Show dataset info
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

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Face Recognition System")
