# AutoML-Model-Trainer
An end-to-end Auto Machine Learning web application built using **Streamlit** and **Scikit-Learn** that allows users to upload a dataset, automatically analyze it, train multiple ML models, evaluate performance, select the best model, and make predictions — all without writing ML code...

---

## 🚀 Features

- 📂 Upload CSV dataset
- 📊 Automated Exploratory Data Analysis (EDA)
- 🧠 Automatic problem type detection (Classification / Regression)
- ⚙ Train multiple ML models automatically
- 📈 Model comparison using relevant metrics
- 🏆 Best model selection
- 🔍 Feature importance visualization
- 🔮 Predict on new user input
- ⬇ Download trained model (`.pkl`)
- 🌐 Interactive web interface using Streamlit

---

## 🛠 Tech Stack

- **Frontend / UI:** Streamlit  
- **Backend:** Python  
- **Machine Learning:** Scikit-Learn  
- **Data Processing:** Pandas, NumPy  

---

## 📂 Project Structure
AutoML-Streamlit/
│
├── web.py # Main Streamlit application
├── best_model.pkl # Saved trained model
├── requirements.txt # Project dependencies
└── README.md # Project documentation

## 🔄 Workflow

1. Upload a cleaned CSV dataset
2. Select the target column
3. App automatically detects problem type
4. Dataset analysis (EDA) is performed
5. Multiple ML models are trained
6. Models are evaluated and compared
7. Best performing model is selected
8. Feature importance is displayed (if applicable)
9. User can test predictions
10. Trained model can be downloaded

---

## 📊 Models Used

### Classification
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier

### Regression
- Random Forest Regressor
- Linear Regression
- Support Vector Regressor (SVR)
- Decision Tree Regressor

---

## 📈 Evaluation Metrics

### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Regression
- MAE
- RMSE
- R² Score

---



### 1️⃣ Clone the repository

git clone https://github.com/your-username/AutoML-Streamlit.git
cd AutoML-Streamlit
---
## 🧪 Usage

1. Upload a CSV file
2. Select the target variable
3. Click Train Models
4. View model performance and best model
5. Enter new feature values for prediction
6. Download the trained model
---
## 📸 Screenshots / Demo

1. Add screenshots of:
2. Dataset upload
3. Model comparison table
4. Feature importance graph
5. Prediction output
---
## 🔮 Future Improvements

1. Hyperparameter tuning
2. Cross-validation
3. SHAP model explainability
4. Support for larger datasets
5. Cloud deployment (Streamlit Cloud)
6. Auto feature scaling and encoding
---
## 👤 Author

Himanshu Singh
🎓 Data Analytics & Machine Learning Enthusiast
📧 Email: himanshusinghchandel5555@gmail.com
---
## ⭐ If you like this project

Give it a ⭐ on GitHub — it motivates me to build more!
---
