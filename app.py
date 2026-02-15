# This is an streamlit app for Multiple Classification Models
import streamlit as st
import numpy as np
import pandas as pd
import io
from model.models import load_data, train_models

st.title("Multiple Classification Models : Interactive app to demonstrate models")

# File uploader widget
uploaded_file = st.file_uploader("Choose input CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.write(
    """
    Choose a model on the sidebar and set input parameters.
    """
)

# Train models
X_train, X_test, y_train, y_test = load_data(df)
scaler, models = train_models(X_train, y_train)

model_names = list(models.keys())

st.sidebar.header('Select Model')
selected_model_name = st.sidebar.selectbox('Classification Model', model_names)
model = models[selected_model_name]

X_test_scaled = scaler.transform(X_test)

prediction = model.predict(X_test)

st.subheader('Prediction')
st.write(prediction)
