from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("xgb_model.pkl")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        input_dict = data.dict()
        print("Incoming data:", input_dict)

        df = pd.DataFrame([input_dict])
        df = pd.get_dummies(df)

        print("One-hot columns:", df.columns.tolist())

        for col in model.get_booster().feature_names:
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = 0

        df = df[model.get_booster().feature_names]

        # Prediction
        prob = model.predict_proba(df)[0][1]
        prediction = "Will Churn" if prob >= 0.5 else "Will Not Churn"

        print("Prediction:", prediction, "| Probability:", prob)

        return {
            "prediction": prediction,
            "churn_probability": float(round(prob, 2))
        }

    except Exception as e:
        print("SERVER ERROR:", e)
        return {"error": str(e)}
