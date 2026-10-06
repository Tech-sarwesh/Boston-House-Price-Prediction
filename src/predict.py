import joblib
import pandas as pd


# 1. Load trained model

model = joblib.load("models/best_model.pkl")


# 2. New house data

new_house = pd.DataFrame({
    "CRIM": [0.1],
    "ZN": [20],
    "INDUS": [5],
    "CHAS": [0],
    "NOX": [0.5],
    "RM": [6.5],
    "AGE": [50],
    "DIS": [5],
    "RAD": [5],
    "TAX": [300],
    "PTRATIO": [15],
    "B": [390],
    "LSTAT": [5]
})


# 3. Make prediction

prediction = model.predict(new_house)


# 4. Display prediction


print("New House Prediction")
print("--------------------")
print("Predicted MEDV:", prediction[0])