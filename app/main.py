import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
)

st.title('Telco Churn prediction app')

st.write("👨‍💼 **Business Use Case**: Reduce customer loss and increase revenue by predicting churn early")
st.write("🔍 **Model**: XGBoost")
st.write("📈 **Recall Score:** 87% ")
st.write("📈 **Precision Score:** 98% ")
