import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

# --------------------------------------------------
# SESSION STATE INITIALIZATION (🔥 VERY IMPORTANT)
# --------------------------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Auto ML Trainer", layout="wide")
st.title(" Auto ML Trainer By Himanshu ")
st.write("Upload a dataset → Analyze → Train → Evaluate → Predict → Download Model")

# --------------------------------------------------
# SIDEBAR – UPLOAD
# --------------------------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset Loaded | Shape: {df.shape}")

    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------
    with st.expander("👀 Dataset Preview"):
        st.dataframe(df.head())

    # --------------------------------------------------
    # TARGET
    # --------------------------------------------------
    st.sidebar.header(" Target Selection")
    target = st.sidebar.selectbox("Select Target Column", df.columns)

    # --------------------------------------------------
    # AUTO PROBLEM TYPE
    # --------------------------------------------------
    if df[target].nunique() <= 10:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    st.info(f"🧠 Detected Problem Type: **{problem_type}**")

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------
    st.subheader("📊 Dataset Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Data Types")
        st.write(df.dtypes)

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    st.write("Target Distribution")
    st.bar_chart(df[target].value_counts())

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include=["object"]):
        X[col] = LabelEncoder().fit_transform(X[col])

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --------------------------------------------------
    # TRAIN BUTTON
    # --------------------------------------------------
    st.subheader("⚙ Model Training")

    if st.button("🚀 Train Models"):
        results = {}
        trained_models = {}

        with st.spinner("Training models..."):
            if problem_type == "Classification":
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "SVM": SVC(),
                    "Decision Tree": DecisionTreeClassifier()
                }

                for name, model in models.items():
                    st.write(f"Training {name}...")
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    results[name] = acc
                    trained_models[name] = model

                best_model_name = max(results, key=results.get)

            else:
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Linear Regression": LinearRegression(),
                    "SVR": SVR(),
                    "Decision Tree": DecisionTreeRegressor()
                }

                for name, model in models.items():
                    st.write(f"Training {name}...")
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    results[name] = mae
                    trained_models[name] = model

                best_model_name = min(results, key=results.get)

        # ✅ STORE IN SESSION STATE
        st.session_state.results = results
        st.session_state.best_model_name = best_model_name
        st.session_state.best_model = trained_models[best_model_name]
        st.session_state.trained = True

        st.success("✅ Training completed successfully!")

    # --------------------------------------------------
    # SHOW RESULTS (ONLY AFTER TRAINING)
    # --------------------------------------------------
    if st.session_state.trained:
        st.subheader("📈 Model Comparison")
        st.dataframe(
            pd.DataFrame.from_dict(
                st.session_state.results,
                orient="index",
                columns=["Score"]
            )
        )

        st.success(f"🏆 Best Model: {st.session_state.best_model_name}")

        best_model = st.session_state.best_model
        preds = best_model.predict(X_test)

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        st.subheader("📊 Model Evaluation")

        if problem_type == "Classification":
            st.dataframe(
                pd.DataFrame(
                    classification_report(y_test, preds, output_dict=True)
                ).transpose()
            )
            st.write("Confusion Matrix")
            st.write(confusion_matrix(y_test, preds))

        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

        # --------------------------------------------------
        # FEATURE IMPORTANCE
        # --------------------------------------------------
        if hasattr(best_model, "feature_importances_"):
            st.subheader("🔍 Feature Importance")
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(fi.set_index("Feature"))

        # --------------------------------------------------
        # DOWNLOAD MODEL
        # --------------------------------------------------
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open("best_model.pkl", "rb") as f:
            st.download_button(
                "⬇ Download Trained Model",
                f,
                file_name="best_model.pkl"
            )

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------
        st.subheader("🔮 Predict on New Data")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = best_model.predict(input_df)
            st.success(f"Prediction: {prediction}")

else:
    st.info("📌 Upload a CSV file to start.")
