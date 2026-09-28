
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/Thyroid-ML-XAI")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "thyroid_xgboost_final.joblib"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "age",
    "sex",
    "on thyroxine",
    "query on thyroxine",
    "on antithyroid medication",
    "sick",
    "pregnant",
    "thyroid surgery",
    "I131 treatment",
    "query hypothyroid",
    "query hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH measured",
    "TSH",
    "T3 measured",
    "TT4 measured",
    "TT4",
    "T4U measured",
    "T4U",
    "FTI measured",
    "FTI"
]

CONTINUOUS_FEATURES = [
    "age",
    "TSH",
    "TT4",
    "T4U",
    "FTI"
]

BINARY_FEATURES = [
    feature
    for feature in FEATURE_NAMES
    if feature not in CONTINUOUS_FEATURES
]

# ColumnTransformer output order
SHAP_FEATURE_NAMES = CONTINUOUS_FEATURES + BINARY_FEATURES


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Thyroid ML XAI API",
    description="Machine Learning and Explainable AI API for thyroid disease dataset analysis",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    age: float

    sex: int
    on_thyroxine: int
    query_on_thyroxine: int
    on_antithyroid_medication: int
    sick: int
    pregnant: int
    thyroid_surgery: int
    I131_treatment: int
    query_hypothyroid: int
    query_hyperthyroid: int
    lithium: int
    goitre: int
    tumor: int
    hypopituitary: int
    psych: int
    TSH_measured: int

    TSH: float

    T3_measured: int

    TT4_measured: int
    TT4: float

    T4U_measured: int
    T4U: float

    FTI_measured: int
    FTI: float


# ============================================================
# CONVERT REQUEST → DATAFRAME
# ============================================================

def request_to_dataframe(request: PredictionRequest):

    data = {
        "age": request.age,
        "sex": request.sex,
        "on thyroxine": request.on_thyroxine,
        "query on thyroxine": request.query_on_thyroxine,
        "on antithyroid medication": request.on_antithyroid_medication,
        "sick": request.sick,
        "pregnant": request.pregnant,
        "thyroid surgery": request.thyroid_surgery,
        "I131 treatment": request.I131_treatment,
        "query hypothyroid": request.query_hypothyroid,
        "query hyperthyroid": request.query_hyperthyroid,
        "lithium": request.lithium,
        "goitre": request.goitre,
        "tumor": request.tumor,
        "hypopituitary": request.hypopituitary,
        "psych": request.psych,
        "TSH measured": request.TSH_measured,
        "TSH": request.TSH,
        "T3 measured": request.T3_measured,
        "TT4 measured": request.TT4_measured,
        "TT4": request.TT4,
        "T4U measured": request.T4U_measured,
        "T4U": request.T4U,
        "FTI measured": request.FTI_measured,
        "FTI": request.FTI
    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_NAMES
    )


# ============================================================
# HEALTH ROUTE
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "project": "Thyroid ML XAI",
        "version": "1.0.0",
        "model": "XGBoost",
        "message": "Thyroid ML XAI API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(FEATURE_NAMES)
    }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Create dataframe
        # ----------------------------------------------------

        df = request_to_dataframe(request)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(model.predict(df)[0])

        probabilities = model.predict_proba(df)[0]

        class_0_probability = float(probabilities[0])
        class_1_probability = float(probabilities[1])


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        try:

            # Get preprocessing transformer
            preprocessor = model.named_steps["preprocessor"]

            # Get classifier
            classifier = model.named_steps["classifier"]

            # Transform input
            transformed = preprocessor.transform(df)

            transformed_array = np.asarray(transformed)

            # SHAP TreeExplainer
            explainer = shap.TreeExplainer(classifier)

            shap_values = explainer.shap_values(
                transformed_array
            )

            shap_array = np.asarray(shap_values)

            # Handle binary classification shape variations
            if shap_array.ndim == 3:

                # Possible shape:
                # samples x features x classes

                shap_row = shap_array[0, :, 1]

            elif shap_array.ndim == 2:

                shap_row = shap_array[0]

            else:

                shap_row = shap_array.reshape(-1)


            # Make sure length matches features
            shap_row = shap_row[:len(SHAP_FEATURE_NAMES)]


            # Values in transformed feature order
            transformed_values = []

            for feature in SHAP_FEATURE_NAMES:

                transformed_values.append(
                    float(df.iloc[0][feature])
                )


            # Build explanation
            explanation = []

            for feature, value, impact in zip(
                SHAP_FEATURE_NAMES,
                transformed_values,
                shap_row
            ):

                explanation.append({
                    "feature": feature,
                    "value": value,
                    "impact": float(impact),
                    "direction": (
                        "increases_class_1"
                        if impact > 0
                        else "decreases_class_1"
                        if impact < 0
                        else "neutral"
                    )
                })


            # Sort by absolute SHAP impact
            explanation.sort(
                key=lambda x: abs(x["impact"]),
                reverse=True
            )


        except Exception as shap_error:

            explanation = []

            print(
                "SHAP explanation error:",
                str(shap_error)
            )


        # ----------------------------------------------------
        # Class label
        # ----------------------------------------------------

        if prediction == 1:

            prediction_label = "Thyroid Disease Predicted"

        else:

            prediction_label = "Thyroid Disease Not Predicted"


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": prediction,

            "prediction_label": prediction_label,

            "probabilities": {

                "class_0": class_0_probability,

                "class_1": class_1_probability

            },

            "model": {

                "name": "XGBoost",

                "version": "Tuned",

                "feature_count": len(FEATURE_NAMES)

            },

            "shap": {

                "available": len(explanation) > 0,

                "features": explanation

            }

        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# RUN INFORMATION
# ============================================================

@app.get("/info")
def info():

    return {

        "project": "Enhancing Thyroid Disease Diagnosis With Machine Learning and Counterfactual Explainable AI",

        "model": "XGBoost",

        "target": "binaryClass",

        "features": FEATURE_NAMES,

        "continuous_features": CONTINUOUS_FEATURES,

        "binary_features": BINARY_FEATURES,

        "feature_count": len(FEATURE_NAMES)

    }
