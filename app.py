from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "thyroid_xgboost_final.joblib"


# ============================================================
# FEATURES
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


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ThyroCare AI API",
    description=(
        "Machine Learning and Explainable AI API "
        "for thyroid disease prediction"
    ),
    version="2.0.0"
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
# REQUEST TO DATAFRAME
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
# MODEL PREDICTION
# ============================================================

def get_prediction(df):

    prediction = int(
        model.predict(df)[0]
    )

    probabilities = model.predict_proba(df)[0]

    return (
        prediction,
        float(probabilities[0]),
        float(probabilities[1])
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap(df):

    try:

        preprocessor = model.named_steps["preprocessor"]

        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(df)

        transformed_array = np.asarray(
            transformed
        )

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = explainer.shap_values(
            transformed_array
        )

        shap_array = np.asarray(
            shap_values
        )

        # Handle different SHAP output shapes
        if shap_array.ndim == 3:

            shap_row = shap_array[0, :, 1]

        elif shap_array.ndim == 2:

            shap_row = shap_array[0]

        else:

            shap_row = shap_array.reshape(-1)

        # Get transformed feature names
        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            feature_names = [
                str(name)
                for name in feature_names
            ]

        except Exception:

            feature_names = FEATURE_NAMES

        explanation = []

        count = min(
            len(shap_row),
            len(feature_names)
        )

        for i in range(count):

            feature_name = feature_names[i]

            # Remove sklearn transformer prefix
            if "__" in feature_name:

                feature_name = feature_name.split(
                    "__",
                    1
                )[1]

            impact = float(
                shap_row[i]
            )

            original_value = None

            # Match feature to original input
            for original_feature in FEATURE_NAMES:

                if (
                    feature_name == original_feature
                    or
                    feature_name.endswith(
                        original_feature
                    )
                ):

                    try:

                        original_value = float(
                            df.iloc[0][
                                original_feature
                            ]
                        )

                    except Exception:

                        original_value = None

                    break

            if impact > 0:

                direction = "increases_class_1"

            elif impact < 0:

                direction = "decreases_class_1"

            else:

                direction = "neutral"

            explanation.append(
                {
                    "feature": feature_name,
                    "value": original_value,
                    "impact": impact,
                    "direction": direction
                }
            )

        explanation.sort(
            key=lambda item: abs(
                item["impact"]
            ),
            reverse=True
        )

        return explanation

    except Exception as error:

        print(
            "SHAP error:",
            str(error)
        )

        return []


# ============================================================
# COUNTERFACTUAL GENERATION
# ============================================================

def generate_counterfactuals(
    original_df,
    original_prediction
):

    candidates = []

    original = original_df.iloc[0].copy()

    # --------------------------------------------------------
    # CONTINUOUS FEATURES
    # --------------------------------------------------------

    continuous_values = {}

    age = float(original["age"])

    continuous_values["age"] = [
        max(1, age - 10),
        max(1, age - 5),
        max(1, age - 2),
        min(100, age + 2),
        min(100, age + 5),
        min(100, age + 10)
    ]

    tsh = float(original["TSH"])

    continuous_values["TSH"] = [
        max(0.01, tsh * 0.25),
        max(0.01, tsh * 0.50),
        max(0.01, tsh * 0.75),
        tsh * 1.25,
        tsh * 1.50,
        tsh * 2.00
    ]

    tt4 = float(original["TT4"])

    continuous_values["TT4"] = [
        max(1, tt4 * 0.60),
        max(1, tt4 * 0.75),
        max(1, tt4 * 0.90),
        tt4 * 1.10,
        tt4 * 1.25,
        tt4 * 1.40
    ]

    t4u = float(original["T4U"])

    continuous_values["T4U"] = [
        max(0.01, t4u * 0.60),
        max(0.01, t4u * 0.80),
        max(0.01, t4u * 0.90),
        t4u * 1.10,
        t4u * 1.20,
        t4u * 1.40
    ]

    fti = float(original["FTI"])

    continuous_values["FTI"] = [
        max(1, fti * 0.60),
        max(1, fti * 0.75),
        max(1, fti * 0.90),
        fti * 1.10,
        fti * 1.25,
        fti * 1.40
    ]

    # --------------------------------------------------------
    # TEST CONTINUOUS CHANGES
    # --------------------------------------------------------

    for feature in CONTINUOUS_FEATURES:

        for new_value in continuous_values[feature]:

            candidate = original.copy()

            candidate[feature] = new_value

            candidate_df = pd.DataFrame(
                [candidate],
                columns=FEATURE_NAMES
            )

            # Ensure binary values are integers
            for binary_feature in BINARY_FEATURES:

                candidate_df[
                    binary_feature
                ] = candidate_df[
                    binary_feature
                ].astype(int)

            try:

                new_prediction = int(
                    model.predict(
                        candidate_df
                    )[0]
                )

                probabilities = (
                    model.predict_proba(
                        candidate_df
                    )[0]
                )

                if new_prediction != original_prediction:

                    original_value = float(
                        original[feature]
                    )

                    distance = abs(
                        new_value -
                        original_value
                    )

                    candidates.append(
                        {
                            "feature": feature,
                            "original_value":
                                original_value,
                            "counterfactual_value":
                                float(new_value),
                            "prediction":
                                new_prediction,
                            "class_0_probability":
                                float(probabilities[0]),
                            "class_1_probability":
                                float(probabilities[1]),
                            "distance":
                                float(distance)
                        }
                    )

            except Exception as error:

                print(
                    "Continuous CF error:",
                    str(error)
                )

    # --------------------------------------------------------
    # TEST BINARY CHANGES
    # --------------------------------------------------------

    for feature in BINARY_FEATURES:

        current_value = int(
            original[feature]
        )

        new_value = 1 - current_value

        candidate = original.copy()

        candidate[feature] = new_value

        candidate_df = pd.DataFrame(
            [candidate],
            columns=FEATURE_NAMES
        )

        for binary_feature in BINARY_FEATURES:

            candidate_df[
                binary_feature
            ] = candidate_df[
                binary_feature
            ].astype(int)

        try:

            new_prediction = int(
                model.predict(
                    candidate_df
                )[0]
            )

            probabilities = (
                model.predict_proba(
                    candidate_df
                )[0]
            )

            if new_prediction != original_prediction:

                candidates.append(
                    {
                        "feature": feature,
                        "original_value":
                            current_value,
                        "counterfactual_value":
                            new_value,
                        "prediction":
                            new_prediction,
                        "class_0_probability":
                            float(probabilities[0]),
                        "class_1_probability":
                            float(probabilities[1]),
                        "distance": 1.0
                    }
                )

        except Exception as error:

            print(
                "Binary CF error:",
                str(error)
            )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["distance"]
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique = []

    seen = set()

    for item in candidates:

        key = (
            item["feature"],
            round(
                float(
                    item["counterfactual_value"]
                ),
                6
            )
        )

        if key not in seen:

            seen.add(key)

            unique.append(item)

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    results = []

    for item in unique[:3]:

        if item["prediction"] == 1:

            label = (
                "Thyroid Disease Predicted"
            )

        else:

            label = (
                "Thyroid Disease Not Predicted"
            )

        results.append(
            {
                "feature":
                    item["feature"],

                "original_value":
                    item["original_value"],

                "counterfactual_value":
                    item["counterfactual_value"],

                "prediction":
                    item["prediction"],

                "prediction_label":
                    label,

                "class_0_probability":
                    item["class_0_probability"],

                "class_1_probability":
                    item["class_1_probability"]
            }
        )

    return results


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "project": "ThyroCare AI",
        "version": "2.0.0",
        "model": "XGBoost",
        "shap": True,
        "counterfactual": True,
        "message": "ThyroCare AI API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(FEATURE_NAMES),
        "shap_available": True,
        "counterfactual_available": True
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        df = request_to_dataframe(
            request
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        (
            prediction,
            class_0_probability,
            class_1_probability
        ) = get_prediction(df)

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if prediction == 1:

            prediction_label = (
                "Thyroid Disease Predicted"
            )

        else:

            prediction_label = (
                "Thyroid Disease Not Predicted"
            )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_features = generate_shap(
            df
        )

        # ----------------------------------------------------
        # COUNTERFACTUAL
        # ----------------------------------------------------

        counterfactual_features = (
            generate_counterfactuals(
                df,
                prediction
            )
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": prediction,

            "prediction_label": prediction_label,

            "probabilities": {

                "class_0":
                    class_0_probability,

                "class_1":
                    class_1_probability

            },

            "model": {

                "name": "XGBoost",

                "version": "Tuned",

                "feature_count":
                    len(FEATURE_NAMES)

            },

            "shap": {

                "available":
                    len(shap_features) > 0,

                "features":
                    shap_features

            },

            "counterfactuals": {

                "available":
                    len(
                        counterfactual_features
                    ) > 0,

                "features":
                    counterfactual_features

            }

        }

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# INFO
# ============================================================

@app.get("/info")
def info():

    return {

        "project": (
            "Enhancing Thyroid Disease Diagnosis "
            "With Machine Learning and "
            "Counterfactual Explainable AI"
        ),

        "model": "XGBoost",

        "target": "binaryClass",

        "features": FEATURE_NAMES,

        "continuous_features":
            CONTINUOUS_FEATURES,

        "binary_features":
            BINARY_FEATURES,

        "feature_count":
            len(FEATURE_NAMES),

        "explainability": {

            "SHAP": True,

            "counterfactual": True

        }

    }
