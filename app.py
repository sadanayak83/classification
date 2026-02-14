import streamlit as st
import numpy as np
from model.models import load_data, train_models

st.title("Iris Flower Classification: Try Multiple Models!")

st.write(
    """
    Choose a model on the sidebar and set your flower measurements.
    """
)

# Train models
X_train, X_test, y_train, y_test = load_data()
scaler, models = train_models(X_train, y_train)

model_names = list(models.keys())

st.sidebar.header('Select Model')
selected_model_name = st.sidebar.selectbox('Classification Model', model_names)
model = models[selected_model_name]

def user_input():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

input_features = user_input()

# Know which models need scaling
scale_needed = selected_model_name in ['Logistic Regression', 'K-Nearest Neighbor']

if scale_needed:
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)
else:
    prediction = model.predict(input_features)

species = ['Setosa', 'Versicolor', 'Virginica']
st.success(f"Predicted class: **{species[prediction[0]]}**")
