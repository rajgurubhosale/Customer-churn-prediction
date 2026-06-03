import streamlit as st

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="centered"
)

# hide sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# hero section
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("#### Predict whether a customer is likely to leave ")

st.divider()

# metrics
col2, col3 = st.columns(2)
with col2:
    st.metric("AUC Score", "93%", "ROC AUC")
with col3:
    st.metric("Recall", "86%", "Minority Class")

st.divider()

# about section
st.markdown("""
###  What is Customer Churn?
Customer churn is when a customer **stops using a service.**  
Predicting churn early helps businesses take action before losing customers.

### ⚙️ How it works?
- Enter customer details on the prediction page
- Model analyzes **14 features**
- Instantly predicts churn probability
""")

st.divider()

# tech stack
st.markdown("### 🛠️ Tech Stack")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("XGBoost")
with c2:
    st.info("MLflow")
with c3:
    st.info("DVC")
with c4:
    st.info("Streamlit")

st.divider()

# CTA button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Start Prediction", use_container_width=True):
        st.switch_page("pages/page_1.py")