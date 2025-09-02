import streamlit as st
from PIL import Image

st.title("📊 Model Performance & Visualizations")

# Confusion Matrix - Stacked
st.subheader("🔷 Confusion Matrix - Stacked Model")
st.image("visuals/conf_matrix_stacked.png", caption="Stacked Model")

# Confusion Matrix - Weighted
st.subheader("🔶 Confusion Matrix - Weighted Model")
st.image("visuals/conf_matrix_weighted.png", caption="Weighted Average Model")

# Accuracy Comparison
st.subheader("📈 Accuracy Comparison")
st.image("visuals/accuracy_comparison.png", caption="Model Accuracy Comparison")

# Classification Reports
st.subheader("📋 Classification Report - Stacked Model")
with open("visuals/stacked_classification_report.txt") as f:
    st.text(f.read())

st.subheader("📋 Classification Report - Weighted Avg Model")
with open("visuals/weighted_classification_report.txt") as f:
    st.text(f.read())
