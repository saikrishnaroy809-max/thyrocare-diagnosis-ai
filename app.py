from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ThyroCare AI API",
    description="Thyroid prediction with XGBoost, SHAP and Counterfactual XAI",
    version="2.1.0",
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
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "thyroid_xgboost_final.joblib"
)


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
    feature
    for feature in FEATURE_NAMES
    if feature not in CONTINUOUS_FEATURES
]


# ============================================================
# LOAD MODEL
# ============================================================

MODEL = None
MODEL_LOAD_ERROR = None

try:

    if not MODEL_PATH.exists():

        MODEL_LOAD_ERROR = (
            f"Model file not found: {MODEL_PATH}"
        )

    else:

        MODEL = joblib.load(MODEL_PATH)

        print(
            "Model loaded successfully:",
            MODEL_PATH,
        )

except Exception as exc:

    MODEL_LOAD_ERROR = str(exc)

    print(
        "Model loading error:",
        MODEL_LOAD_ERROR,
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    # Allows frontend field aliases such as
    # TSH_measured and I131_treatment.

    model_config = ConfigDict(
        populate_by_name=True
    )

    age: float
    sex: int

    on_thyroxine: int
    query_on_thyroxine: int
    on_antithyroid_medication: int
    sick: int
    pregnant: int
    thyroid_surgery: int

    i131_treatment: int = Field(
        alias="I131_treatment"
    )

    query_hypothyroid: int
    query_hyperthyroid: int
    lithium: int
    goitre: int
    tumor: int
    hypopituitary: int
    psych: int

    tsh_measured: int = Field(
        alias="TSH_measured"
    )

    TSH: float

    t3_measured: int = Field(
        alias="T3_measured"
    )

    tt4_measured: int = Field(
        alias="TT4_measured"
    )

    TT4: float

    t4u_measured: int = Field(
        alias="T4U_measured"
    )

    T4U: float

    fti_measured: int = Field(
        alias="FTI_measured"
    )

    FTI: float


# ============================================================
# REQUEST → DATAFRAME
# ============================================================

def request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:

    data = {

        "age": request.age,

        "sex": request.sex,

        "on thyroxine":
            request.on_thyroxine,

        "query on thyroxine":
            request.query_on_thyroxine,

        "on antithyroid medication":
            request.on_antithyroid_medication,

        "sick":
            request.sick,

        "pregnant":
            request.pregnant,

        "thyroid surgery":
            request.thyroid_surgery,

        "I131 treatment":
            request.i131_treatment,

        "query hypothyroid":
            request.query_hypothyroid,

        "query hyperthyroid":
            request.query_hyperthyroid,

        "lithium":
            request.lithium,

        "goitre":
            request.goitre,

        "tumor":
            request.tumor,

        "hypopituitary":
            request.hypopituitary,

        "psych":
            request.psych,

        "TSH measured":
            request.tsh_measured,

        "TSH":
            request.TSH,

        "T3 measured":
            request.t3_measured,

        "TT4 measured":
            request.tt4_measured,

        "TT4":
            request.TT4,

        "T4U measured":
            request.t4u_measured,

        "T4U":
            request.T4U,

        "FTI measured":
            request.fti_measured,

        "FTI":
            request.FTI,
    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_NAMES,
    )


# ============================================================
# PREDICTION
# ============================================================

def get_prediction(
    df: pd.DataFrame,
):

    if MODEL is None:

        raise RuntimeError(
            MODEL_LOAD_ERROR
            or "Model is not loaded."
        )

    prediction = int(
        MODEL.predict(df)[0]
    )

    probabilities = (
        MODEL.predict_proba(df)[0]
    )

    class_0 = float(
        probabilities[0]
    )

    class_1 = float(
        probabilities[1]
    )

    if prediction == 1:

        label = (
            "Thyroid Disease Predicted"
        )

    else:

        label = (
            "Thyroid Disease Not Predicted"
        )

    return (
        prediction,
        label,
        class_0,
        class_1,
    )


# ============================================================
# SHAP
# ============================================================

def generate_shap(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    try:

        if MODEL is None:

            return {
                "available": False,
                "features": [],
            }

        classifier = MODEL

        if hasattr(
            MODEL,
            "named_steps",
        ):

            if (
                "classifier"
                in MODEL.named_steps
            ):

                classifier = (
                    MODEL.named_steps[
                        "classifier"
                    ]
                )

            elif (
                "model"
                in MODEL.named_steps
            ):

                classifier = (
                    MODEL.named_steps[
                        "model"
                    ]
                )

        transformed_df = df

        if (
            hasattr(
                MODEL,
                "named_steps",
            )
            and
            "preprocessor"
            in MODEL.named_steps
        ):

            preprocessor = (
                MODEL.named_steps[
                    "preprocessor"
                ]
            )

            try:

                transformed_df = (
                    preprocessor.transform(df)
                )

            except Exception:

                transformed_df = df

        explainer = (
            shap.TreeExplainer(
                classifier
            )
        )

        shap_values = (
            explainer.shap_values(
                transformed_df
            )
        )

        if isinstance(
            shap_values,
            list,
        ):

            if len(shap_values) > 1:

                values = np.asarray(
                    shap_values[1]
                )[0]

            else:

                values = np.asarray(
                    shap_values[0]
                )[0]

        else:

            values_array = np.asarray(
                shap_values
            )

            if values_array.ndim == 3:

                values = (
                    values_array[
                        0,
                        :,
                        -1
                    ]
                )

            elif values_array.ndim == 2:

                values = (
                    values_array[0]
                )

            else:

                values = (
                    values_array.flatten()
                )

        values = np.asarray(
            values
        ).flatten()

        feature_names = None

        if (
            hasattr(
                MODEL,
                "named_steps",
            )
            and
            "preprocessor"
            in MODEL.named_steps
        ):

            try:

                feature_names = list(
                    MODEL.named_steps[
                        "preprocessor"
                    ].get_feature_names_out()
                )

            except Exception:

                feature_names = None

        if (
            not feature_names
            or len(feature_names)
            != len(values)
        ):

            feature_names = (
                FEATURE_NAMES[
                    :len(values)
                ]
            )

        results = []

        for name, value in zip(
            feature_names,
            values,
        ):

            clean_name = str(name)

            if "__" in clean_name:

                clean_name = (
                    clean_name.split(
                        "__",
                        1,
                    )[1]
                )

            results.append(
                {
                    "feature": clean_name,
                    "impact": float(
                        value
                    ),
                }
            )

        results.sort(
            key=lambda item:
                abs(item["impact"]),
            reverse=True,
        )

        return {
            "available": True,
            "features": results[:10],
        }

    except Exception as exc:

        print(
            "SHAP error:",
            exc,
        )

        return {
            "available": False,
            "features": [],
            "error": str(exc),
        }


# ============================================================
# COUNTERFACTUAL VALUES
# ============================================================

def create_candidate_values(
    feature: str,
    current_value: float,
) -> List[float]:

    current = float(
        current_value
    )

    if feature == "age":

        values = [
            current - 15,
            current - 10,
            current - 5,
            current + 5,
            current + 10,
            current + 15,
        ]

        values = [
            max(
                1.0,
                min(
                    100.0,
                    value,
                ),
            )
            for value in values
        ]

    elif feature == "TSH":

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

    elif feature == "TT4":

        values = [
            40.0,
            50.0,
            60.0,
            75.0,
            90.0,
            100.0,
            110.0,
            125.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ]

    elif feature == "T4U":

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

    elif feature == "FTI":

        values = [
            40.0,
            50.0,
            60.0,
            75.0,
            90.0,
            100.0,
            110.0,
            125.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ]

    else:

        values = [
            0
            if int(round(current)) == 1
            else 1
        ]

    cleaned = []

    for value in values:

        value = float(value)

        if abs(
            value - current
        ) < 1e-9:

            continue

        if not any(
            abs(
                value - existing
            ) < 1e-9
            for existing in cleaned
        ):

            cleaned.append(value)

    return cleaned


# ============================================================
# COUNTERFACTUAL DISTANCE
# ============================================================

def feature_distance(
    original: pd.Series,
    candidate: pd.Series,
    feature: str,
) -> float:

    original_value = float(
        original[feature]
    )

    candidate_value = float(
        candidate[feature]
    )

    if feature in CONTINUOUS_FEATURES:

        if feature == "age":

            scale = 20.0

        elif feature == "TSH":

            scale = max(
                abs(original_value),
                2.0,
            )

        elif feature == "TT4":

            scale = max(
                abs(original_value),
                50.0,
            )

        elif feature == "T4U":

            scale = max(
                abs(original_value),
                0.5,
            )

        elif feature == "FTI":

            scale = max(
                abs(original_value),
                50.0,
            )

        else:

            scale = 1.0

        return abs(
            candidate_value
            - original_value
        ) / scale

    return (
        1.0
        if candidate_value
        != original_value
        else 0.0
    )


def total_distance(
    original: pd.Series,
    candidate: pd.Series,
) -> float:

    total = 0.0

    for feature in FEATURE_NAMES:

        total += feature_distance(
            original,
            candidate,
            feature,
        )

    return float(total)


# ============================================================
# MULTI-FEATURE COUNTERFACTUAL SEARCH
# ============================================================

def generate_counterfactuals(
    original_df: pd.DataFrame,
    original_prediction: int,
    shap_result: Dict[str, Any],
) -> Dict[str, Any]:

    try:

        original_row = (
            original_df.iloc[0].copy()
        )

        # Important continuous features
        search_features = [
            "TSH",
            "TT4",
            "T4U",
            "FTI",
            "age",
        ]

        # Add important SHAP features.
        for item in shap_result.get(
            "features",
            [],
        ):

            feature = item.get(
                "feature"
            )

            if not feature:
                continue

            if "__" in feature:

                feature = feature.split(
                    "__",
                    1,
                )[1]

            if feature not in FEATURE_NAMES:
                continue

            if feature in MEASURED_FEATURES:
                continue

            if feature not in search_features:

                search_features.append(
                    feature
                )

        search_features = (
            search_features[:10]
        )

        candidate_values = {}

        for feature in search_features:

            candidate_values[
                feature
            ] = create_candidate_values(
                feature,
                original_row[
                    feature
                ],
            )

        # Initial beam.
        beam = [
            {
                "row":
                    original_row.copy(),

                "changed": [],

                "distance": 0.0,
            }
        ]

        found = []

        MAX_DEPTH = 4

        BEAM_WIDTH = 30

        target_class = (
            1 - int(
                original_prediction
            )
        )

        for depth in range(
            1,
            MAX_DEPTH + 1,
        ):

            generated = []

            for state in beam:

                current_row = (
                    state["row"]
                )

                changed_features = set(
                    state["changed"]
                )

                for feature in search_features:

                    if feature in changed_features:
                        continue

                    for value in (
                        candidate_values.get(
                            feature,
                            [],
                        )
                    ):

                        new_row = (
                            current_row.copy()
                        )

                        if (
                            feature
                            in BINARY_FEATURES
                        ):

                            new_row[
                                feature
                            ] = int(
                                round(
                                    value
                                )
                            )

                        else:

                            new_row[
                                feature
                            ] = float(value)

                        changed = list(
                            state["changed"]
                        )

                        changed.append(
                            feature
                        )

                        generated.append(
                            {
                                "row":
                                    new_row,

                                "changed":
                                    changed,

                                "distance":
                                    total_distance(
                                        original_row,
                                        new_row,
                                    ),
                            }
                        )

            if not generated:
                break

            # Remove duplicate states.
            unique = {}

            for state in generated:

                key = tuple(
                    (
                        feature,
                        round(
                            float(
                                state[
                                    "row"
                                ][feature]
                            ),
                            6,
                        ),
                    )
                    for feature
                    in FEATURE_NAMES
                )

                if key not in unique:

                    unique[key] = state

            generated = list(
                unique.values()
            )

            # Batch prediction.
            batch_df = pd.DataFrame(
                [
                    state["row"]
                    for state in generated
                ],
                columns=FEATURE_NAMES,
            )

            predictions = MODEL.predict(
                batch_df
            )

            probabilities = (
                MODEL.predict_proba(
                    batch_df
                )
            )

            scored = []

            for index, state in enumerate(
                generated
            ):

                prediction = int(
                    predictions[index]
                )

                class_0 = float(
                    probabilities[
                        index
                    ][0]
                )

                class_1 = float(
                    probabilities[
                        index
                    ][1]
                )

                target_probability = (
                    class_1
                    if target_class == 1
                    else class_0
                )

                score = (
                    target_probability
                    - (
                        0.08
                        * state[
                            "distance"
                        ]
                    )
                )

                state[
                    "prediction"
                ] = prediction

                state[
                    "class_0_probability"
                ] = class_0

                state[
                    "class_1_probability"
                ] = class_1

                state[
                    "target_probability"
                ] = target_probability

                state[
                    "score"
                ] = score

                scored.append(
                    state
                )

                if (
                    prediction
                    == target_class
                ):

                    found.append(
                        state
                    )

            scored.sort(
                key=lambda item: (
                    item["score"],
                    -item["distance"],
                ),
                reverse=True,
            )

            beam = scored[
                :BEAM_WIDTH
            ]

            if found:
                break

        # No counterfactual.
        if not found:

            return {
                "available": False,
                "features": [],
                "scenarios": [],
                "message": (
                    "No counterfactual was found "
                    "within the tested feature ranges."
                ),
            }

        # Sort found scenarios.
        found.sort(
            key=lambda item: (
                len(item["changed"]),
                item["distance"],
                -item[
                    "target_probability"
                ],
            )
        )

        best_scenarios = found[:3]

        scenarios = []

        for scenario_number, scenario in enumerate(
            best_scenarios,
            start=1,
        ):

            changes = []

            for feature in scenario[
                "changed"
            ]:

                original_value = (
                    original_row[
                        feature
                    ]
                )

                new_value = (
                    scenario["row"][
                        feature
                    ]
                )

                if (
                    feature
                    in BINARY_FEATURES
                ):

                    original_value = int(
                        round(
                            float(
                                original_value
                            )
                        )
                    )

                    new_value = int(
                        round(
                            float(
                                new_value
                            )
                        )
                    )

                else:

                    original_value = float(
                        original_value
                    )

                    new_value = float(
                        new_value
                    )

                changes.append(
                    {
                        "feature": feature,
                        "original_value":
                            original_value,
                        "counterfactual_value":
                            new_value,
                    }
                )

            target_prediction = int(
                scenario[
                    "prediction"
                ]
            )

            if target_prediction == 1:

                target_label = (
                    "Thyroid Disease Predicted"
                )

            else:

                target_label = (
                    "Thyroid Disease Not Predicted"
                )

            scenarios.append(
                {
                    "scenario":
                        scenario_number,

                    "changes":
                        changes,

                    "prediction":
                        target_prediction,

                    "prediction_label":
                        target_label,

                    "class_0_probability":
                        round(
                            scenario[
                                "class_0_probability"
                            ],
                            6,
                        ),

                    "class_1_probability":
                        round(
                            scenario[
                                "class_1_probability"
                            ],
                            6,
                        ),

                    "distance":
                        round(
                            float(
                                scenario[
                                    "distance"
                                ]
                            ),
                            6,
                        ),
                }
            )

        best = scenarios[0]

        return {
            "available": True,

            # Compatible with current frontend.
            "features":
                best["changes"],

            "scenarios":
                scenarios,

            "target_prediction":
                best["prediction"],

            "target_label":
                best["prediction_label"],
        }

    except Exception as exc:

        print(
            "Counterfactual error:",
            exc,
        )

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "error": str(exc),
        }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "ThyroCare AI API",
        "version": "2.1.0",
        "model_loaded":
            MODEL is not None,
        "model": "XGBoost",
        "features":
            len(FEATURE_NAMES),
        "endpoints": [
            "/",
            "/health",
            "/info",
            "/predict",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded":
            MODEL is not None,
        "features":
            len(FEATURE_NAMES),
        "shap_available": True,
        "counterfactual_available": True,
    }


# ============================================================
# INFO
# ============================================================

@app.get("/info")
def info():

    return {
        "model": "XGBoost",
        "version": "Tuned",
        "feature_count":
            len(FEATURE_NAMES),
        "features":
            FEATURE_NAMES,
        "continuous_features":
            CONTINUOUS_FEATURES,
        "model_loaded":
            MODEL is not None,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    if MODEL is None:

        raise HTTPException(
            status_code=500,
            detail=(
                MODEL_LOAD_ERROR
                or "Model is not loaded."
            ),
        )

    try:

        # Convert request.
        df = request_to_dataframe(
            request
        )

        # Prediction.
        (
            prediction,
            prediction_label,
            class_0_probability,
            class_1_probability,
        ) = get_prediction(df)

        # SHAP.
        shap_result = generate_shap(
            df
        )

        # Counterfactual.
        counterfactual_result = (
            generate_counterfactuals(
                df,
                prediction,
                shap_result,
            )
        )

        return {

            "success": True,

            "prediction":
                prediction,

            "prediction_label":
                prediction_label,

            "probabilities": {

                "class_0":
                    round(
                        class_0_probability,
                        6,
                    ),

                "class_1":
                    round(
                        class_1_probability,
                        6,
                    ),
            },

            "model": {

                "name":
                    "XGBoost",

                "version":
                    "Tuned",

                "feature_count":
                    len(FEATURE_NAMES),
            },

            "shap":
                shap_result,

            "counterfactuals":
                counterfactual_result,
        }

    except Exception as exc:

        print(
            "Prediction error:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
    )
