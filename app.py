# ============================================================
# ThyroCare AI - FastAPI Backend
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
    version="2.2.0",
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
# MODEL
# ============================================================

model = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL_LOAD_ERROR = str(e)


# ============================================================
# TRAINING FEATURES
# IMPORTANT:
# These names MUST match the original dataset.
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
# PYDANTIC REQUEST MODEL
#
# IMPORTANT:
# These names intentionally match the React frontend.
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

    # EXACT FRONTEND NAME
    I131_treatment: int

    query_hypothyroid: int
    query_hyperthyroid: int

    lithium: int
    goitre: int
    tumor: int
    hypopituitary: int
    psych: int

    # EXACT FRONTEND NAMES
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

def request_to_dataframe(req: PredictionRequest) -> pd.DataFrame:

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

    df = pd.DataFrame(
        [data],
        columns=FEATURE_NAMES
    )

    return df


# ============================================================
# BASIC PREDICTION
# ============================================================

def get_prediction(df: pd.DataFrame):

    if model is None:
        raise RuntimeError(
            f"Model could not be loaded: {MODEL_LOAD_ERROR}"
        )

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    prediction = int(prediction)

    class_0_probability = float(probabilities[0])
    class_1_probability = float(probabilities[1])

    if prediction == 1:
        label = "Thyroid Disease Predicted"
    else:
        label = "Thyroid Disease Not Predicted"

    return {
        "prediction": prediction,
        "label": label,
        "probabilities": {
            "class_0": class_0_probability,
            "class_1": class_1_probability,
        },
    }


# ============================================================
# SHAP
# ============================================================

def generate_shap(df: pd.DataFrame):

    if model is None:
        return {
            "available": False,
            "features": [],
            "error": "Model is not loaded",
        }

    try:

        import shap

        # Extract classifier from sklearn pipeline
        classifier = model

        if hasattr(model, "named_steps"):
            if "classifier" in model.named_steps:
                classifier = model.named_steps["classifier"]

        # Transform dataframe if preprocessing exists
        transformed = df

        feature_names = FEATURE_NAMES.copy()

        if hasattr(model, "named_steps"):

            if "preprocessor" in model.named_steps:

                preprocessor = model.named_steps["preprocessor"]

                transformed = preprocessor.transform(df)

                try:
                    transformed_names = (
                        preprocessor.get_feature_names_out()
                    )

                    feature_names = list(transformed_names)

                except Exception:
                    feature_names = FEATURE_NAMES.copy()

        # Create SHAP explainer
        explainer = shap.TreeExplainer(classifier)

        shap_values = explainer.shap_values(transformed)

        # Handle binary classification output
        if isinstance(shap_values, list):

            if len(shap_values) > 1:
                values = np.asarray(shap_values[1])[0]
            else:
                values = np.asarray(shap_values[0])[0]

        else:

            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, -1]

            elif values.ndim == 2:
                values = values[0]

            else:
                values = values.reshape(-1)

        values = values.astype(float)

        # Make sure names and values have same length
        count = min(len(values), len(feature_names))

        values = values[:count]
        feature_names = feature_names[:count]

        result = []

        for name, value in zip(feature_names, values):

            clean_name = str(name)

            # Make preprocessing names easier to understand
            if "__" in clean_name:
                clean_name = clean_name.split("__")[-1]

            result.append(
                {
                    "feature": clean_name,
                    "impact": float(value),
                    "abs_impact": float(abs(value)),
                }
            )

        # Sort by importance
        result.sort(
            key=lambda x: x["abs_impact"],
            reverse=True
        )

        return {
            "available": True,
            "features": result[:15],
        }

    except Exception as e:

        return {
            "available": False,
            "features": [],
            "error": str(e),
        }


# ============================================================
# COUNTERFACTUAL HELPERS
# ============================================================

def safe_predict(df: pd.DataFrame):

    try:
        prediction = model.predict(df)
        return np.asarray(prediction).astype(int)

    except Exception:
        return None


def create_candidate_values(
    feature: str,
    current_value: Any
) -> List[Any]:

    try:

        current = float(current_value)

    except Exception:

        return []


    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if feature == "age":

        values = [
            current - 15,
            current - 10,
            current - 5,
            current + 5,
            current + 10,
            current + 15,
        ]

        return [
            float(round(max(1, min(100, x)), 2))
            for x in values
            if abs(x - current) > 0.001
        ]


    # --------------------------------------------------------
    # TSH
    # --------------------------------------------------------

    if feature == "TSH":

        values = [
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            7.0,
            10.0,
            15.0,
            20.0,
        ]

        return [
            x for x in values
            if abs(x - current) > 0.001
        ]


    # --------------------------------------------------------
    # TT4
    # --------------------------------------------------------

    if feature == "TT4":

        values = [
            40,
            50,
            60,
            75,
            90,
            100,
            110,
            125,
            140,
            160,
            180,
            200,
        ]

        return [
            float(x) for x in values
            if abs(x - current) > 0.001
        ]


    # --------------------------------------------------------
    # T4U
    # --------------------------------------------------------

    if feature == "T4U":

        values = [
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.5,
        ]

        return [
            float(x) for x in values
            if abs(x - current) > 0.001
        ]


    # --------------------------------------------------------
    # FTI
    # --------------------------------------------------------

    if feature == "FTI":

        values = [
            40,
            50,
            60,
            75,
            90,
            100,
            110,
            125,
            140,
            160,
            180,
            200,
        ]

        return [
            float(x) for x in values
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
    new_value: Any
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
    modified: pd.DataFrame
) -> float:

    total = 0.0

    for feature in FEATURE_NAMES:

        old_value = original.iloc[0][feature]
        new_value = modified.iloc[0][feature]

        if old_value != new_value:

            total += feature_distance(
                feature,
                old_value,
                new_value
            )

    return float(total)


# ============================================================
# COUNTERFACTUAL GENERATION
# ============================================================

def generate_counterfactuals(
    original_df: pd.DataFrame,
    original_prediction: int,
    shap_result: Dict[str, Any]
):

    if model is None:

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "message": "Model is not loaded.",
        }


    # --------------------------------------------------------
    # Determine important features from SHAP
    # --------------------------------------------------------

    important_features = []

    try:

        shap_features = shap_result.get(
            "features",
            []
        )

        for item in shap_features:

            name = item.get("feature")

            if name in FEATURE_NAMES:
                important_features.append(name)

    except Exception:
        pass


    # --------------------------------------------------------
    # Core continuous features
    # --------------------------------------------------------

    search_features = [
        "TSH",
        "TT4",
        "T4U",
        "FTI",
        "age",
    ]


    # --------------------------------------------------------
    # Add important binary features
    # --------------------------------------------------------

    for feature in important_features:

        if feature not in search_features:

            if feature not in MEASURED_FEATURES:

                if feature in FEATURE_NAMES:

                    search_features.append(feature)

        if len(search_features) >= 10:
            break


    # --------------------------------------------------------
    # Create candidate values
    # --------------------------------------------------------

    candidates = {}

    for feature in search_features:

        current_value = original_df.iloc[0][feature]

        values = create_candidate_values(
            feature,
            current_value
        )

        if values:
            candidates[feature] = values


    # --------------------------------------------------------
    # First search: one feature at a time
    # --------------------------------------------------------

    scenarios = []

    for feature in search_features:

        if feature not in candidates:
            continue

        for value in candidates[feature]:

            candidate_df = original_df.copy()

            candidate_df.at[
                0,
                feature
            ] = value

            prediction = safe_predict(candidate_df)

            if prediction is None:
                continue

            new_prediction = int(prediction[0])

            if new_prediction != original_prediction:

                distance = total_distance(
                    original_df,
                    candidate_df
                )

                scenarios.append(
                    {
                        "features": {
                            feature: value
                        },
                        "distance": distance,
                        "prediction": new_prediction,
                    }
                )


    # --------------------------------------------------------
    # Multi-feature beam search
    # --------------------------------------------------------

    beam = [
        (
            original_df.copy(),
            []
        )
    ]

    MAX_DEPTH = 4
    BEAM_WIDTH = 30


    for depth in range(MAX_DEPTH):

        next_beam = []

        for current_df, changes in beam:

            for feature in search_features:

                # Avoid changing same feature repeatedly
                changed_features = [
                    item["feature"]
                    for item in changes
                ]

                if feature in changed_features:
                    continue

                if feature not in candidates:
                    continue

                for value in candidates[feature]:

                    candidate_df = current_df.copy()

                    candidate_df.at[
                        0,
                        feature
                    ] = value

                    new_changes = changes + [
                        {
                            "feature": feature,
                            "old_value": current_df.iloc[0][feature],
                            "new_value": value,
                        }
                    ]

                    prediction = safe_predict(
                        candidate_df
                    )

                    if prediction is None:
                        continue

                    new_prediction = int(
                        prediction[0]
                    )

                    distance = total_distance(
                        original_df,
                        candidate_df
                    )

                    if new_prediction != original_prediction:

                        scenario_features = {}

                        for change in new_changes:

                            scenario_features[
                                change["feature"]
                            ] = change["new_value"]

                        scenarios.append(
                            {
                                "features": scenario_features,
                                "distance": distance,
                                "prediction": new_prediction,
                            }
                        )

                    else:

                        next_beam.append(
                            (
                                candidate_df,
                                new_changes
                            )
                        )


        # Keep closest candidates
        next_beam.sort(
            key=lambda item: total_distance(
                original_df,
                item[0]
            )
        )

        beam = next_beam[:BEAM_WIDTH]

        if not beam:
            break


    # --------------------------------------------------------
    # Remove duplicate scenarios
    # --------------------------------------------------------

    unique = {}

    for scenario in scenarios:

        key = tuple(
            sorted(
                scenario["features"].items()
            )
        )

        if key not in unique:

            unique[key] = scenario

        else:

            if (
                scenario["distance"]
                < unique[key]["distance"]
            ):

                unique[key] = scenario


    scenarios = list(unique.values())


    # --------------------------------------------------------
    # Sort by smallest change
    # --------------------------------------------------------

    scenarios.sort(
        key=lambda x: x["distance"]
    )


    # --------------------------------------------------------
    # Return no-result response
    # --------------------------------------------------------

    if not scenarios:

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "message": (
                "No counterfactual scenario "
                "was found within the tested feature ranges."
            ),
        }


    # --------------------------------------------------------
    # Top scenarios
    # --------------------------------------------------------

    top_scenarios = scenarios[:3]


    best = top_scenarios[0]


    # --------------------------------------------------------
    # Format changes for frontend
    # --------------------------------------------------------

    formatted_changes = []

    for feature, new_value in best["features"].items():

        old_value = original_df.iloc[0][feature]

        formatted_changes.append(
            {
                "feature": feature,
                "old_value": (
                    float(old_value)
                    if isinstance(
                        old_value,
                        (np.integer, np.floating)
                    )
                    else old_value
                ),
                "new_value": (
                    float(new_value)
                    if isinstance(
                        new_value,
                        (np.integer, np.floating)
                    )
                    else new_value
                ),
            }
        )


    return {
        "available": True,

        # Compatible with existing App.jsx
        "features": formatted_changes,

        "scenarios": top_scenarios,

        "message": (
            "A model sensitivity scenario was found."
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
        "version": "2.2.0",
        "model_loaded": model is not None,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "unhealthy",
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
def predict(req: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Convert request to dataframe
        # ----------------------------------------------------

        df = request_to_dataframe(req)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        result = get_prediction(df)


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_result = generate_shap(df)


        # ----------------------------------------------------
        # Counterfactual
        # ----------------------------------------------------

        counterfactual_result = generate_counterfactuals(
            df,
            result["prediction"],
            shap_result
        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {
            "success": True,

            "prediction": result["prediction"],

            "label": result["label"],

            "probabilities": result["probabilities"],

            "model": "XGBoost",

            "shap": shap_result,

            "counterfactuals": counterfactual_result,
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        )
