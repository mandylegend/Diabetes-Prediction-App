import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import os

df = pd.read_csv("diabetes_prediction_dataset.csv")

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

le = LabelEncoder()
for column in X.columns:
    if X[column].dtype == "object":
        X[column] = le.fit_transform(X[column])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "diabetes_model.pkl")
st.markdown(
    "<h1 style='text-align: center;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Enter patient details to predict diabetes risk</p>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

.stButton > button {
    background-color: #22c55e;
    color: black;
    font-weight: bold;
}

label {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# st.markdown("""
# <style>
# .stApp {
#     background-image: url("https://images.unsplash.com/photo-1582719478250-c89cae4dc85b");
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
# }
# </style>
# """, unsafe_allow_html=True)


st.divider()

# sidebar colors


st.markdown("""
<style>
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)



def user_input_features():
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.slider("Age", 0, 100, 30)
    hypertension = st.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.selectbox("Heart Disease", ("Yes", "No"))
    smoking_history = st.selectbox(
        "Smoking History", ("Never", "Formerly", "Currently"))
    bmi = st.slider("BMI", 0.0, 50.0, 25.0)
    if bmi == 0.0:
        pass
    HbA1c_level = st.slider("HbA1c Level", 0.0, 15.0, 5.0)
    blood_glucose_level = st.slider(
        "Blood Glucose Level", 0.0, 300.0, 100.0)
    data = {
        "gender": 1 if gender == "Male" else 0,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": 0 if smoking_history == "Never" else (1 if smoking_history == "Formerly" else 2),
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()


def load_model():
    return joblib.load("diabetes_model.pkl")


model = load_model()

if st.button("Submit"):
    prediction = model.predict(input_df)
    perctengae = model.predict_proba(input_df)[0][1] * 100
    st.write(f"Diabetes Risk: {perctengae:.2f}%")

    input_df["diabetes"] = model.predict(input_df)
    if os.path.exists("user_data.csv"):
        input_df.to_csv("user_data.csv", mode="a", header=False, index=False)
    else:
        input_df.to_csv("user_data.csv", index=False)   

    st.write("Result:",
             "Diabetic" if prediction[0] == 1 else "Not Diabetic")

    if prediction[0] == 1:
        st.warning(
            "You are at risk of diabetes. Please consult a healthcare professional.")

        st.markdown("### What is Diabetes?")
        st.markdown("""
            Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). 
            It occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces.
            This leads to elevated blood sugar levels, which can cause various health complications if left unmanaged.
        """)

        st.markdown("### Common Symptoms of Diabetes:")
        st.markdown("""
            - Increased thirst and frequent urination.
            - Unexplained weight loss.
            - Fatigue and weakness.
            - Blurred vision.
            - Slow-healing sores or frequent infections.
            - Tingling or numbness in hands or feet.
            - Persistent hunger.""")

    else:
        st.success(
            "You are not at risk of diabetes. Keep maintaining a healthy lifestyle!")

        st.markdown("### Tips to Reduce Diabetes Risk:")
        st.markdown("""
            - Maintain a healthy weight through balanced diet and regular exercise.
            - Limit sugary foods and beverages.
            - Choose whole grains over refined grains.
            - Stay active with at least 150 minutes of moderate exercise per week.
            - Avoid smoking and limit alcohol consumption.
            - Regularly monitor blood sugar levels and consult your doctor for personalized advice.
            -  Manage stress through mindfulness, meditation, or hobbies you enjoy.
                    """)
    

    

# motivations = [
#     "Every healthy choice counts.",
#     "Your future self will thank you.",
#     "Health today, strength tomorrow.",
#     "One good decision can change everything.",
#     "Stay consistent, stay healthy."
# ]
# st.success(random.choice(motivations))
