# This is an streamlit app for Multiple Classification Models
import streamlit as st
import numpy as np
import pandas as pd
import io
from model.models import load_data, train_models

st.title("Multiple Classification Models : Interactive app to demonstrate models")

#Code starts here to download test csv file
CSV_FILE_PATH = 'data/winequality.csv'
DOWNLOAD_FILE_NAME = 'winequality.csv'
MIME_TYPE = 'text/csv'

st.write("Download Existing CSV File")

st.write("Click the button below to download the pre-existing test CSV file.")

try:
    with open(CSV_FILE_PATH, "rb") as f:
        csv_bytes = f.read()
    
    st.download_button(
        label="📥 Download CSV File",
        data=csv_bytes,
        file_name=DOWNLOAD_FILE_NAME,
        mime=MIME_TYPE
    )

except FileNotFoundError:
    st.error(f"Error: Not able to download the CSV file.")
    st.info("Please make sure the CSV file exists.")


#Code starts here to upload test csv file
# File uploader widget
uploaded_file = st.file_uploader("Choose input CSV file", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.write(
    """
    Choose a model on the sidebar.
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

metrics = []
metrics.append(evaluate_model(name, model, X_test, y_test, prediction))
st.subheader('Evaluation Metrics')
st.write(metrics)

confusion_matrix = []
confusion_matrix.append(get_confusion_metrics(name, y_test, prediction))
st.subheader('Confusion Matrix')
st.write(confusion_matrix)
