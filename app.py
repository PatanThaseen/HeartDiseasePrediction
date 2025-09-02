import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("model/stacked_model.pkl")

# Load dataset
data = pd.read_csv("data/heart.csv")

# Split X and y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Prediction", "Visualizations"])

# ---------- Prediction Page ----------
if app_mode == "Prediction":
    st.title("Heart Disease Prediction")

    st.write("Enter the patient data below:")

    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG Result", [0, 1])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [3, 6, 7])

    if st.button("Predict"):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=X.columns)

        prediction = model.predict(input_data)[0]
        result = "Heart Disease" if prediction == 1 else "No Heart Disease"
        st.subheader("Prediction Result:")
        st.success(result)

# ---------- Visualizations Page ----------
elif app_mode == "Visualizations":
    st.title("Model Performance Visualizations")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Recreate models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    # Predictions
    rf_pred = rf.predict_proba(X_test)
    svm_pred = svm.predict_proba(X_test)
    knn_pred = knn.predict_proba(X_test)

    weights = [0.4, 0.3, 0.3]
    weighted_pred = (weights[0] * rf_pred + weights[1] * svm_pred + weights[2] * knn_pred).argmax(axis=1)

    # Stacked Model Accuracy
    y_pred = model.predict(X_test)
    stacked_acc = accuracy_score(y_test, y_pred) * 100
    weighted_acc = accuracy_score(y_test, weighted_pred) * 100

    # Confusion Matrix - Stacked
    st.subheader("🔷 Confusion Matrix - Stacked Model")
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # Classification Report - Stacked
    st.subheader("📋 Classification Report - Stacked Model")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix - Weighted Model
    st.subheader("🔶 Confusion Matrix - Weighted Avg Model")
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, weighted_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # Classification Report - Weighted Model
    st.subheader("📋 Classification Report - Weighted Avg Model")
    st.text(classification_report(y_test, weighted_pred))

    # Accuracy Comparison Bar Chart
    st.subheader("📈 Accuracy Comparison")
    fig3, ax3 = plt.subplots()
    ax3.bar(["Stacked Model", "Weighted Avg Model"], [stacked_acc, weighted_acc], color=['mediumpurple', 'plum'])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Accuracy (%)")
    st.pyplot(fig3)
