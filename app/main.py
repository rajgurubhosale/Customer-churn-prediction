import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
)

 
st.title('Churn prediction app')
