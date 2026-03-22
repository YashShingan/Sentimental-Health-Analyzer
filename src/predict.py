import joblib
import pandas as pd

model = joblib.load("../models/mental_health_model.pkl")

def predict_risk(input_data):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)

    return prediction