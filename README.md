# 🧠 Face Recognition System

A machine learning project that performs **face recognition** using **PCA (Principal Component Analysis)** for dimensionality reduction and **SVM (Support Vector Machine)** for classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

---

## 📋 Overview

This project demonstrates a complete machine learning pipeline for face recognition:

1. **Data Loading** - Uses LFW (Labeled Faces in the Wild) dataset
2. **Preprocessing** - PCA reduces ~3000 pixel features to 150 components
3. **Model Training** - SVM with RBF kernel for classification
4. **Optimization** - GridSearchCV for hyperparameter tuning
5. **Evaluation** - Accuracy, Confusion Matrix, Classification Report

---

## 🚀 Features

- ✅ Face recognition using PCA + SVM pipeline
- ✅ Hyperparameter optimization with GridSearchCV
- ✅ Interactive Streamlit dashboard
- ✅ Confusion Matrix visualization
- ✅ Pie chart for prediction distribution
- ✅ Sample prediction display

---

## 📁 Project Structure

```
ML project/
├── Faces.py              # Main training script
├── app.py                # Streamlit dashboard
├── README.md             # This file
├── confusion_matrix.png  # Generated confusion matrix
└── prediction_samples.png # Generated sample predictions
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/face-recognition-system.git
cd face-recognition-system

# Install dependencies
pip install numpy matplotlib seaborn scikit-learn streamlit
```

---

## 💻 Usage

### Run the main script

```bash
python Faces.py
```

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

---

## 🔬 How It Works

### 1. PCA (Principal Component Analysis)

- **Problem**: Each face image has ~3000 pixels (features)
- **Solution**: PCA reduces to 150 principal components
- **Benefit**: Faster training, less overfitting

### 2. SVM (Support Vector Machine)

- **Kernel**: RBF (Radial Basis Function)
- **Why**: Works well with high-dimensional data
- **Class Weight**: Balanced to handle imbalanced classes

### 3. GridSearchCV

- **Parameters tuned**: `C` (regularization), `gamma` (kernel coefficient)
- **Method**: 5-fold cross-validation
- **Result**: Finds optimal hyperparameters automatically

---

## 📊 Results

| Metric     | Value       |
| ---------- | ----------- |
| Accuracy   | ~85-90%     |
| Best C     | 5-10        |
| Best Gamma | 0.001-0.005 |

---

## 📈 Visualizations

The project generates:

- **Confusion Matrix**: Shows which faces are confused with others
- **Pie Chart**: Distribution of predictions
- **Sample Predictions**: Visual display of correct/incorrect predictions

---

## 🔮 Future Improvements

- [ ] Add real-time webcam face recognition
- [ ] Implement deep learning (CNN) for better accuracy
- [ ] Add liveness detection (anti-spoofing)
- [ ] Deploy as web API

---

## 📚 Technologies Used

- **Python 3.8+**
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - ML algorithms (PCA, SVM, GridSearchCV)
- **Streamlit** - Interactive dashboard

---

## 👤 Author

**Your Name**  
GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
