# ============================================================
# THYROCARE AI - FASTAPI BACKEND
# Thyroid Disease Prediction + SHAP + Counterfactual AI
# ============================================================

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="ThyroCare AI API",
    description="AI-powered thyroid disease prediction with Explainable AI",
    version="3.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "thyroid_xgboost_final.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

model = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)
    print("ThyroCare model loaded successfully.")

except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print("MODEL LOAD ERROR:", MODEL_LOAD_ERROR)


# ============================================================
# ORIGINAL TRAINING FEATURES
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
    "FTI",
]


CONTINUOUS_FEATURES = [
    "age",
    "TSH",
    "TT4",
    "T4U",
    "FTI",
]


MEASURED_FEATURES = [
    "TSH measured",
    "T3 measured",
    "TT4 measured",
    "T4U measured",
    "FTI measured",
]


BINARY_FEATURES = [
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
]


# ============================================================
# REQUEST MODEL
#
# IMPORTANT:
# These names match the React frontend.
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
# REQUEST -> DATAFRAME
# ============================================================

def request_to_dataframe(
    req: PredictionRequest,
) -> pd.DataFrame:

    data = {
        "age": req.age,
        "sex": req.sex,

        "on thyroxine": req.on_thyroxine,
        "query on thyroxine": req.query_on_thyroxine,
        "on antithyroid medication": req.on_antithyroid_medication,

        "sick": req.sick,
        "pregnant": req.pregnant,

        "thyroid surgery": req.thyroid_surgery,
        "I131 treatment": req.I131_treatment,

        "query hypothyroid": req.query_hypothyroid,
        "query hyperthyroid": req.query_hyperthyroid,

        "lithium": req.lithium,
        "goitre": req.goitre,
        "tumor": req.tumor,
        "hypopituitary": req.hypopituitary,
        "psych": req.psych,

        "TSH measured": req.TSH_measured,
        "TSH": req.TSH,

        "T3 measured": req.T3_measured,

        "TT4 measured": req.TT4_measured,
        "TT4": req.TT4,

        "T4U measured": req.T4U_measured,
        "T4U": req.T4U,

        "FTI measured": req.FTI_measured,
        "FTI": req.FTI,
    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_NAMES,
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_prediction(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if model is None:
        raise RuntimeError(
            f"Model could not be loaded: {MODEL_LOAD_ERROR}"
        )

    prediction = int(model.predict(df)[0])

    probabilities = model.predict_proba(df)[0]

    class_0_probability = float(probabilities[0])
    class_1_probability = float(probabilities[1])

    if prediction == 1:
        label = "Thyroid Disease Predicted"
    else:
        label = "Thyroid Disease Not Predicted"

    return {
        "prediction": prediction,
        "label": label,
        "class_0_probability": class_0_probability,
        "class_1_probability": class_1_probability,
    }


# ============================================================
# SHAP EXPLANATION
#
# Optimized for one patient.
# ============================================================

def generate_shap(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if model is None:

        return {
            "available": False,
            "features": [],
            "error": "Model is not loaded.",
        }

    try:

        import shap

        classifier = model

        transformed = df

        feature_names = FEATURE_NAMES.copy()

        # ----------------------------------------------------
        # Handle sklearn pipeline
        # ----------------------------------------------------

        if hasattr(model, "named_steps"):

            if "classifier" in model.named_steps:
                classifier = model.named_steps["classifier"]

            if "preprocessor" in model.named_steps:

                preprocessor = model.named_steps["preprocessor"]

                transformed = preprocessor.transform(df)

                try:

                    feature_names = list(
                        preprocessor.get_feature_names_out()
                    )

                except Exception:

                    feature_names = FEATURE_NAMES.copy()

        # ----------------------------------------------------
        # TreeExplainer
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(classifier)

        shap_values = explainer.shap_values(
            transformed
        )

        # ----------------------------------------------------
        # Handle SHAP output formats
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            if len(shap_values) > 1:
                values = np.asarray(
                    shap_values[1]
                )[0]

            else:
                values = np.asarray(
                    shap_values[0]
                )[0]

        else:

            values = np.asarray(shap_values)

            if values.ndim == 3:

                values = values[0, :, -1]

            elif values.ndim == 2:

                values = values[0]

            else:

                values = values.reshape(-1)

        values = values.astype(float)

        count = min(
            len(values),
            len(feature_names),
        )

        result = []

        for i in range(count):

            name = str(feature_names[i])

            value = float(values[i])

            # Clean preprocessing prefixes
            if "__" in name:
                name = name.split("__")[-1]

            result.append(
                {
                    "feature": name,
                    "value": value,
                    "impact": value,
                    "abs_impact": abs(value),
                }
            )

        result.sort(
            key=lambda x: x["abs_impact"],
            reverse=True,
        )

        return {
            "available": True,
            "features": result[:15],
        }

    except Exception as e:

        print("SHAP ERROR:", str(e))

        return {
            "available": False,
            "features": [],
            "error": str(e),
        }


# ============================================================
# SAFE MODEL PREDICTION
# ============================================================

def safe_predict(
    df: pd.DataFrame,
) -> int | None:

    try:

        prediction = model.predict(df)

        return int(prediction[0])

    except Exception as e:

        print("Counterfactual prediction error:", e)

        return None


# ============================================================
# SAFE PROBABILITIES
# ============================================================

def get_probabilities(
    df: pd.DataFrame,
) -> Dict[str, float]:

    try:

        probabilities = model.predict_proba(df)[0]

        return {
            "class_0_probability": float(
                probabilities[0]
            ),
            "class_1_probability": float(
                probabilities[1]
            ),
        }

    except Exception:

        return {
            "class_0_probability": 0.0,
            "class_1_probability": 0.0,
        }


# ============================================================
# CANDIDATE VALUES
#
# IMPORTANT:
# Small candidate sets prevent long-running requests.
# ============================================================

def create_candidate_values(
    feature: str,
    current_value: Any,
) -> List[Any]:

    try:
        current = float(current_value)

    except Exception:
        return []

    # --------------------------------------------------------
    # TSH
    # --------------------------------------------------------

    if feature == "TSH":

        values = [
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            7.0,
            10.0,
            15.0,
        ]

        return [
            x
            for x in values
            if abs(x - current) > 0.001
        ]

    # --------------------------------------------------------
    # TT4
    # --------------------------------------------------------

    if feature == "TT4":

        values = [
            60,
            80,
            100,
            110,
            125,
            150,
            180,
        ]

        return [
            float(x)
            for x in values
            if abs(x - current) > 0.001
        ]

    # --------------------------------------------------------
    # T4U
    # --------------------------------------------------------

    if feature == "T4U":

        values = [
            0.5,
            0.7,
            0.9,
            1.0,
            1.1,
            1.3,
        ]

        return [
            float(x)
            for x in values
            if abs(x - current) > 0.001
        ]

    # --------------------------------------------------------
    # FTI
    # --------------------------------------------------------

    if feature == "FTI":

        values = [
            60,
            80,
            100,
            110,
            125,
            150,
            180,
        ]

        return [
            float(x)
            for x in values
            if abs(x - current) > 0.001
        ]

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if feature == "age":

        values = [
            max(1, current - 10),
            max(1, current - 5),
            current + 5,
            min(100, current + 10),
        ]

        return [
            round(float(x), 2)
            for x in values
            if abs(x - current) > 0.001
        ]

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    if feature in BINARY_FEATURES:

        if int(round(current)) == 0:
            return [1]

        return [0]

    return []


# ============================================================
# DISTANCE
# ============================================================

def feature_distance(
    feature: str,
    old_value: Any,
    new_value: Any,
) -> float:

    try:

        old = float(old_value)
        new = float(new_value)

    except Exception:

        return 1.0

    if feature == "age":
        return abs(new - old) / 20.0

    if feature == "TSH":
        return abs(new - old) / 10.0

    if feature == "TT4":
        return abs(new - old) / 100.0

    if feature == "T4U":
        return abs(new - old)

    if feature == "FTI":
        return abs(new - old) / 100.0

    return abs(new - old)


def total_distance(
    original: pd.DataFrame,
    modified: pd.DataFrame,
) -> float:

    total = 0.0

    for feature in FEATURE_NAMES:

        old_value = original.iloc[0][feature]

        new_value = modified.iloc[0][feature]

        if old_value != new_value:

            total += feature_distance(
                feature,
                old_value,
                new_value,
            )

    return float(total)


# ============================================================
# FAST COUNTERFACTUAL GENERATION
#
# IMPORTANT:
# This replaces the previous expensive beam search.
#
# We test only a limited number of meaningful changes.
# ============================================================

def generate_counterfactuals(
    original_df: pd.DataFrame,
    original_prediction: int,
    shap_result: Dict[str, Any],
) -> Dict[str, Any]:

    if model is None:

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "message": "Model is not loaded.",
        }

    # --------------------------------------------------------
    # Start with medically relevant continuous features
    # --------------------------------------------------------

    search_features = [
        "TSH",
        "TT4",
        "T4U",
        "FTI",
        "age",
    ]

    # --------------------------------------------------------
    # Add important SHAP features
    # --------------------------------------------------------

    try:

        shap_features = shap_result.get(
            "features",
            []
        )

        for item in shap_features:

            feature = item.get("feature")

            if feature in FEATURE_NAMES:

                if feature not in search_features:

                    if feature not in MEASURED_FEATURES:

                        search_features.append(feature)

            # Keep search deliberately small
            if len(search_features) >= 8:
                break

    except Exception:
        pass

    # --------------------------------------------------------
    # Limit number of features
    # --------------------------------------------------------

    search_features = search_features[:8]

    scenarios = []

    # --------------------------------------------------------
    # Test individual feature changes only.
    #
    # Maximum roughly:
    # 8 features × 7 values = 56 predictions
    #
    # This is dramatically faster than the old beam search.
    # --------------------------------------------------------

    for feature in search_features:

        current_value = original_df.iloc[0][feature]

        candidate_values = create_candidate_values(
            feature,
            current_value,
        )

        for new_value in candidate_values:

            candidate_df = original_df.copy()

            candidate_df.at[
                0,
                feature,
            ] = new_value

            new_prediction = safe_predict(
                candidate_df
            )

            if new_prediction is None:
                continue

            # Only keep changes that alter prediction
            if new_prediction != original_prediction:

                probabilities = get_probabilities(
                    candidate_df
                )

                distance = total_distance(
                    original_df,
                    candidate_df,
                )

                scenarios.append(
                    {
                        "changes": [
                            {
                                "feature": feature,
                                "original_value": (
                                    float(current_value)
                                    if isinstance(
                                        current_value,
                                        (np.integer, np.floating),
                                    )
                                    else current_value
                                ),
                                "counterfactual_value": (
                                    float(new_value)
                                    if isinstance(
                                        new_value,
                                        (np.integer, np.floating),
                                    )
                                    else new_value
                                ),
                            }
                        ],
                        "prediction": new_prediction,
                        "prediction_label": (
                            "Thyroid Disease Predicted"
                            if new_prediction == 1
                            else
                            "Thyroid Disease Not Predicted"
                        ),
                        "class_0_probability": probabilities[
                            "class_0_probability"
                        ],
                        "class_1_probability": probabilities[
                            "class_1_probability"
                        ],
                        "distance": distance,
                    }
                )

    # --------------------------------------------------------
    # Sort closest changes first
    # --------------------------------------------------------

    scenarios.sort(
        key=lambda x: x["distance"]
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}

    for scenario in scenarios:

        change = scenario["changes"][0]

        key = (
            change["feature"],
            change["counterfactual_value"],
        )

        if key not in unique:
            unique[key] = scenario

    scenarios = list(unique.values())

    scenarios.sort(
        key=lambda x: x["distance"]
    )

    # --------------------------------------------------------
    # No counterfactual found
    # --------------------------------------------------------

    if not scenarios:

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "message": (
                "No counterfactual scenario was found "
                "within the tested feature ranges."
            ),
        }

    # --------------------------------------------------------
    # Maximum 3 scenarios
    # --------------------------------------------------------

    scenarios = scenarios[:3]

    # --------------------------------------------------------
    # Best scenario changes
    # --------------------------------------------------------

    best_changes = scenarios[0]["changes"]

    return {
        "available": True,

        "features": [
            {
                "feature": item["feature"],
                "original_value": item[
                    "original_value"
                ],
                "counterfactual_value": item[
                    "counterfactual_value"
                ],
            }
            for item in best_changes
        ],

        "scenarios": scenarios,

        "target_prediction": scenarios[0][
            "prediction"
        ],

        "target_label": scenarios[0][
            "prediction_label"
        ],

        "message": (
            "Counterfactual scenarios generated successfully."
        ),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "ThyroCare AI API",
        "status": "online",
        "version": "3.0.0",
        "model_loaded": model is not None,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if model is not None
            else "unhealthy"
        ),
        "model_loaded": model is not None,
        "features": len(FEATURE_NAMES),
        "shap_available": True,
        "counterfactual_available": True,
    }


# ============================================================
# INFO
# ============================================================

@app.get("/info")
def info():

    return {
        "name": "ThyroCare AI",
        "model": "XGBoost",
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "explainability": {
            "shap": True,
            "counterfactual": True,
        },
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    req: PredictionRequest,
):

    try:

        print("Prediction request received.")

        # ----------------------------------------------------
        # Convert request to dataframe
        # ----------------------------------------------------

        df = request_to_dataframe(req)

        print("Input dataframe created.")

        # ----------------------------------------------------
        # Basic prediction
        # ----------------------------------------------------

        result = get_prediction(df)

        print(
            "Prediction completed:",
            result["prediction"],
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_result = generate_shap(df)

        print("SHAP completed.")

        # ----------------------------------------------------
        # Counterfactual
        # ----------------------------------------------------

        counterfactual_result = (
            generate_counterfactuals(
                df,
                result["prediction"],
                shap_result,
            )
        )

        print("Counterfactual completed.")

        # ----------------------------------------------------
        # IMPORTANT:
        # Return fields compatible with App.jsx
        # ----------------------------------------------------

        return {
            "success": True,

            "prediction": result[
                "prediction"
            ],

            "prediction_label": result[
                "label"
            ],

            "label": result[
                "label"
            ],

            "model": "XGBoost",

            "class_0_probability": result[
                "class_0_probability"
            ],

            "class_1_probability": result[
                "class_1_probability"
            ],

            "probabilities": {
                "class_0": result[
                    "class_0_probability"
                ],
                "class_1": result[
                    "class_1_probability"
                ],
            },

            "shap_values": shap_result[
                "features"
            ],

            "shap": shap_result,

            "counterfactuals": counterfactual_result,
        }

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        )
