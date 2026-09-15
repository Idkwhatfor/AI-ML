from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained SVM model
model = joblib.load("diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the form
    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    diabetes_pedigree = float(request.form["diabetes_pedigree"])
    age = float(request.form["age"])

    # Arrange values in the same order used during training
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get confidence if the SVM supports probability prediction
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        # Probability of the predicted class
        predicted_class_index = list(model.classes_).index(prediction)
        confidence = probabilities[predicted_class_index] * 100

    else:
        # SVM was not trained with probability=True
        confidence = None

    # Result
    if prediction == 1:
        result = "Diabetes"
    else:
        result = "Not Detected"

    # Patient report
    patient_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "Skin Thickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Diabetes Pedigree Function": diabetes_pedigree,
        "Age": age
    }

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        patient_data=patient_data
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
