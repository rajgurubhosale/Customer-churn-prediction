import sys
sys.path.append(r'D:\churn prediction')

import pandas as pd
import streamlit as st 
from src.feature_engineering import drop_multicollinear_features,add_total_charges,group_state
import pickle

st.title('🎯 Batch Prediction')
test = st.file_uploader(r"$\textsf{\Large Enter Csv File}$")

if st.button('Make Prediction'):
    df = pd.read_csv(test)
    df = add_total_charges(df)
    df = group_state(df)
    df = drop_multicollinear_features(df)
    model = pickle.load(open(r"models/final_model.pkl",'rb'))
    pred = model.predict(df)
    df['Model prediction'] = pred
    st.write(df['Model prediction'].value_counts())
    st.dataframe(df)


#drop_state
#addd_total_charges
#group_staten
#remove multicollinear features
#tnf = data+ num_cols,cat_cols
#load model
#predict
