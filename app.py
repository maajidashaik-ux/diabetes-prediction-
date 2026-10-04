import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def get_model():
    data = pd.read_csv("diabetes.csv")
    x = data.drop(columns="Outcome")
    y = data["Outcome"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x, y)
    return model


model = get_model()

st.title("Diabetes Prediction")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=columns,
    )
    result = model.predict(data)[0]
    if result == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
