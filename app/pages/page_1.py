import streamlit as st
import pandas as pd
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from src.utils.mlflow_utils import setup_mlflow

# load model once
@st.cache_resource
def load_model():
    setup_mlflow()
    pipeline = mlflow.sklearn.load_model("models:/xgb_classifier@production-model")
    return pipeline

# hide sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

if st.button("← Back to Home"):
    st.switch_page("main.py")

st.title("🔍 Predict Customer Churn")
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    account_length         = st.number_input("Account Length",          min_value=0,   value=100)
    area_code              = st.number_input("Area Code",               min_value=0,   value=415)
    international_plan     = st.selectbox("International Plan",         [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    voice_mail_plan        = st.selectbox("Voice Mail Plan",            [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    total_day_calls        = st.number_input("Total Day Calls",         min_value=0,   value=100)
    total_eve_minutes      = st.number_input("Total Eve Minutes",       min_value=0.0, value=200.0)
    total_eve_calls        = st.number_input("Total Eve Calls",         min_value=0,   value=100)

with col2:
    total_night_minutes    = st.number_input("Total Night Minutes",     min_value=0.0, value=200.0)
    total_night_calls      = st.number_input("Total Night Calls",       min_value=0,   value=100)
    total_intl_minutes     = st.number_input("Total Intl Minutes",      min_value=0.0, value=10.0)
    total_intl_calls       = st.number_input("Total Intl Calls",        min_value=0,   value=3)
    customer_service_calls = st.number_input("Customer Service Calls",  min_value=0,   value=1)
    total_calls            = st.number_input("Total Calls",             min_value=0,   value=300)
    total_charge           = st.number_input("Total Charge",            min_value=0.0, value=40.0)

if st.button("🔍 Predict", use_container_width=True):

    input_df = pd.DataFrame([{
        "Account length":         account_length,
        "Area code":              area_code,
        "International plan":     international_plan,
        "Voice mail plan":        voice_mail_plan,
        "Total day calls":        total_day_calls,
        "Total eve minutes":      total_eve_minutes,
        "Total eve calls":        total_eve_calls,
        "Total night minutes":    total_night_minutes,
        "Total night calls":      total_night_calls,
        "Total intl minutes":     total_intl_minutes,
        "Total intl calls":       total_intl_calls,
        "Customer service calls": customer_service_calls,
        "total_calls":            total_calls,
        "total_charge":           total_charge
    }])

    pipeline = load_model()
    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()

    # prediction result
    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to CHURN! (Probability: {probability:.1%})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {probability:.1%})")

    # --- SHAP explanation ---
    st.divider()
    st.subheader("🔎 Why did the model predict this?")

    # extract xgboost model from pipeline
    xgb_model = pipeline.named_steps['model']

    # transform input using pipeline transformer
    tnf = pipeline.named_steps['tnf']
    input_transformed = tnf.transform(input_df)

    # shap explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_transformed)

    # feature names after transformation
    feature_names = (
        input_df.select_dtypes(include=['int64','float64']).columns.tolist() +
        input_df.select_dtypes(include=['object']).columns.tolist()
    )

    # waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_transformed[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)

    # top factors table
    st.subheader("📋 Top Factors")
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    }).sort_values("Impact", key=abs, ascending=False).head(5)

    shap_df["Direction"] = shap_df["Impact"].apply(
        lambda x: "🔴 Increases Churn" if x > 0 else "🟢 Decreases Churn"
    )
    st.dataframe(shap_df[["Feature", "Direction", "Impact"]], use_container_width=True)