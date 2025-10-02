import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ========================
# 1. Load dataset (via uploader)
# ========================
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person has **diabetes** based on medical data.")

uploaded_file = st.file_uploader("📂 Upload your diabetes.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========================
    # 2. Train & Save Model
    # ========================
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "diabetes_model.joblib")

    # Model accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ========================
    # 3. Streamlit UI
    # ========================
    st.sidebar.header("Enter Patient Data")

    # User input form
    def user_input():
        Pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.sidebar.slider("Glucose", 0, 200, 120)
        BloodPressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
        SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
        Insulin = st.sidebar.slider("Insulin", 0, 900, 80)
        BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.sidebar.slider("Age", 18, 100, 30)

        data = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }
        return pd.DataFrame(data, index=[0])

    # Get input
    input_df = user_input()

    st.subheader("🔍 Patient Data")
    st.write(input_df)

    # ========================
    # 4. Prediction
    # ========================
    loaded_model = joblib.load("diabetes_model.joblib")
    prediction = loaded_model.predict(input_df)[0]
    prediction_proba = loaded_model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The model predicts: **Diabetes** (probability: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"✅ The model predicts: **No Diabetes** (probability: {prediction_proba[0]*100:.2f}%)")

    st.info(f"Model Accuracy on Test Data: {acc*100:.2f}%")

else:
    st.warning("👆 Please upload the **diabetes.csv** file to continue.")
