"""
============================================================
THYROCARE AI - FASTAPI BACKEND
============================================================

Thyroid Disease Prediction + SHAP + Counterfactual AI

Admin Dataset Upload
Admin Preprocessing
Admin Model Training
Model Comparison
Persistent Admin State
Persistent Prediction History

============================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="ThyroCare AI API",
    description=(
        "AI-powered thyroid disease prediction "
        "with Explainable AI and Admin ML Management"
    ),
    version="7.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://thyrocare-diagnosis-ai.vercel.app",
        "https://thyrocare-diagnosis-ai-q2ln.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "thyroid_xgboost_final.joblib"
)

DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UPLOADED_DATASET_PATH = (
    DATASET_DIR / "uploaded_dataset.csv"
)

PREPROCESSED_DATASET_PATH = (
    DATASET_DIR / "preprocessed_dataset.csv"
)

TRAINING_RESULTS_PATH = (
    DATASET_DIR / "training_results.json"
)

TRAINING_MODELS_PATH = (
    DATASET_DIR / "training_models.joblib"
)

ADMIN_STATE_PATH = (
    DATASET_DIR / "admin_state.json"
)

PREDICTION_HISTORY_PATH = (
    DATASET_DIR / "prediction_history.json"
)


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

model = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)

    print("============================================")
    print("ThyroCare production model loaded.")
    print("Model:", MODEL_PATH)
    print("============================================")

except Exception as e:
    MODEL_LOAD_ERROR = str(e)

    print("============================================")
    print("MODEL LOAD ERROR")
    print(MODEL_LOAD_ERROR)
    print("============================================")


# ============================================================
# ORIGINAL 25 FEATURES
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
# ADMIN STATE
# ============================================================

DATASET = None
DATASET_NAME = None
DATASET_TARGET = None

PREPROCESSED_DATASET = None
PREPROCESSING_INFO = None

TRAINING_RESULTS = []
TRAINING_MODELS = {}
TRAINING_INFO = None

PREDICTION_HISTORY = []


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

TARGET_COLUMN_CANDIDATES = [
    "binaryClass",
    "binaryclass",
    "target",
    "Target",
    "class",
    "Class",
    "label",
    "Label",
    "diagnosis",
    "Diagnosis",
    "thyroid",
    "Thyroid",
    "thyroid_disease",
    "thyroid disease",
    "disease",
    "Disease",
]


def detect_target_column(
    dataframe: pd.DataFrame,
):
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in TARGET_COLUMN_CANDIDATES:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if len(dataframe.columns) > 0:
        return dataframe.columns[-1]

    return None


# ============================================================
# PERSISTENCE
# ============================================================

def save_admin_state():
    state = {
        "dataset_name": DATASET_NAME,
        "dataset_target": (
            str(DATASET_TARGET)
            if DATASET_TARGET is not None
            else None
        ),
        "preprocessing_info": PREPROCESSING_INFO,
        "training_info": TRAINING_INFO,
    }

    try:
        with open(
            ADMIN_STATE_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
                default=str,
            )

        print("Admin state saved.")

    except Exception as e:
        print(
            "ADMIN STATE SAVE ERROR:",
            str(e),
        )


def save_training_state():
    try:
        with open(
            TRAINING_RESULTS_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "results": TRAINING_RESULTS,
                    "training": TRAINING_INFO,
                },
                file,
                indent=2,
                default=str,
            )

        if TRAINING_MODELS:
            joblib.dump(
                TRAINING_MODELS,
                TRAINING_MODELS_PATH,
            )

        print("Training state saved.")

    except Exception as e:
        print(
            "TRAINING STATE SAVE ERROR:",
            str(e),
        )


def save_prediction_history():
    try:
        with open(
            PREDICTION_HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                PREDICTION_HISTORY,
                file,
                indent=2,
                default=str,
            )

        print(
            "Prediction history saved:",
            len(PREDICTION_HISTORY),
        )

    except Exception as e:
        print(
            "PREDICTION HISTORY SAVE ERROR:",
            str(e),
        )


def load_admin_state():

    global DATASET
    global DATASET_NAME
    global DATASET_TARGET

    global PREPROCESSED_DATASET
    global PREPROCESSING_INFO

    global TRAINING_RESULTS
    global TRAINING_MODELS
    global TRAINING_INFO

    global PREDICTION_HISTORY

    print("============================================")
    print("Restoring saved admin state...")
    print("============================================")

    # --------------------------------------------------------
    # Uploaded dataset
    # --------------------------------------------------------

    if UPLOADED_DATASET_PATH.exists():
        try:
            DATASET = pd.read_csv(
                UPLOADED_DATASET_PATH
            )

            print(
                "Uploaded dataset restored:",
                DATASET.shape,
            )

        except Exception as e:
            print(
                "Dataset restore error:",
                str(e),
            )

    # --------------------------------------------------------
    # Preprocessed dataset
    # --------------------------------------------------------

    if PREPROCESSED_DATASET_PATH.exists():
        try:
            PREPROCESSED_DATASET = pd.read_csv(
                PREPROCESSED_DATASET_PATH
            )

            print(
                "Preprocessed dataset restored:",
                PREPROCESSED_DATASET.shape,
            )

        except Exception as e:
            print(
                "Preprocessed dataset restore error:",
                str(e),
            )

    # --------------------------------------------------------
    # Admin metadata
    # --------------------------------------------------------

    if ADMIN_STATE_PATH.exists():
        try:
            with open(
                ADMIN_STATE_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                state = json.load(file)

            DATASET_NAME = state.get(
                "dataset_name"
            )

            DATASET_TARGET = state.get(
                "dataset_target"
            )

            PREPROCESSING_INFO = state.get(
                "preprocessing_info"
            )

            TRAINING_INFO = state.get(
                "training_info"
            )

            print("Admin metadata restored.")

        except Exception as e:
            print(
                "Admin metadata restore error:",
                str(e),
            )

    # --------------------------------------------------------
    # Detect target again if necessary
    # --------------------------------------------------------

    if (
        DATASET is not None
        and DATASET_TARGET is None
    ):
        DATASET_TARGET = detect_target_column(
            DATASET
        )

    # --------------------------------------------------------
    # Training results
    # --------------------------------------------------------

    if TRAINING_RESULTS_PATH.exists():
        try:
            with open(
                TRAINING_RESULTS_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                training_state = json.load(file)

            TRAINING_RESULTS = training_state.get(
                "results",
                [],
            )

            if TRAINING_INFO is None:
                TRAINING_INFO = training_state.get(
                    "training"
                )

            print(
                "Training results restored:",
                len(TRAINING_RESULTS),
            )

        except Exception as e:
            print(
                "Training results restore error:",
                str(e),
            )

    # --------------------------------------------------------
    # Trained models
    # --------------------------------------------------------

    if TRAINING_MODELS_PATH.exists():
        try:
            TRAINING_MODELS = joblib.load(
                TRAINING_MODELS_PATH
            )

            print(
                "Trained models restored:",
                len(TRAINING_MODELS),
            )

        except Exception as e:
            print(
                "Trained model restore error:",
                str(e),
            )

    # --------------------------------------------------------
    # Prediction history
    # --------------------------------------------------------

    if PREDICTION_HISTORY_PATH.exists():

        try:
            with open(
                PREDICTION_HISTORY_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                saved_history = json.load(file)

            if isinstance(
                saved_history,
                list,
            ):
                PREDICTION_HISTORY = (
                    saved_history
                )
            else:
                PREDICTION_HISTORY = []

            print(
                "Prediction history restored:",
                len(PREDICTION_HISTORY),
            )

        except Exception as e:

            print(
                "Prediction history restore error:",
                str(e),
            )

            PREDICTION_HISTORY = []

    else:
        PREDICTION_HISTORY = []

    print("Admin state restoration completed.")
    print("============================================")


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
# REQUEST -> DATAFRAME
# ============================================================

def request_to_dataframe(
    req: PredictionRequest,
) -> pd.DataFrame:

    data = {
        "age": req.age,
        "sex": req.sex,

        "on thyroxine": req.on_thyroxine,
        "query on thyroxine":
            req.query_on_thyroxine,

        "on antithyroid medication":
            req.on_antithyroid_medication,

        "sick": req.sick,
        "pregnant": req.pregnant,

        "thyroid surgery":
            req.thyroid_surgery,

        "I131 treatment":
            req.I131_treatment,

        "query hypothyroid":
            req.query_hypothyroid,

        "query hyperthyroid":
            req.query_hyperthyroid,

        "lithium": req.lithium,
        "goitre": req.goitre,
        "tumor": req.tumor,
        "hypopituitary": req.hypopituitary,
        "psych": req.psych,

        "TSH measured":
            req.TSH_measured,

        "TSH": req.TSH,

        "T3 measured":
            req.T3_measured,

        "TT4 measured":
            req.TT4_measured,

        "TT4": req.TT4,

        "T4U measured":
            req.T4U_measured,

        "T4U": req.T4U,

        "FTI measured":
            req.FTI_measured,

        "FTI": req.FTI,
    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_NAMES,
    )


# ============================================================
# PRODUCTION PREDICTION
# ============================================================

def get_prediction(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if model is None:
        raise RuntimeError(
            "Production model could not be loaded: "
            f"{MODEL_LOAD_ERROR}"
        )

    prediction = int(
        model.predict(df)[0]
    )

    probabilities = model.predict_proba(
        df
    )[0]

    class_0_probability = float(
        probabilities[0]
    )

    class_1_probability = float(
        probabilities[1]
    )

    if prediction == 1:
        label = "Thyroid Disease Predicted"
    else:
        label = "Thyroid Disease Not Predicted"

    return {
        "prediction": prediction,
        "label": label,
        "class_0_probability":
            class_0_probability,
        "class_1_probability":
            class_1_probability,
    }


# ============================================================
# SHAP
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

        # Pipeline support
        if hasattr(model, "named_steps"):

            if "classifier" in model.named_steps:
                classifier = model.named_steps[
                    "classifier"
                ]

            if "preprocessor" in model.named_steps:

                preprocessor = model.named_steps[
                    "preprocessor"
                ]

                transformed = preprocessor.transform(
                    df
                )

                try:
                    feature_names = list(
                        preprocessor.get_feature_names_out()
                    )

                except Exception:
                    feature_names = (
                        FEATURE_NAMES.copy()
                    )

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = explainer.shap_values(
            transformed
        )

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

            values = np.asarray(
                shap_values
            )

            if values.ndim == 3:
                values = values[
                    0,
                    :,
                    -1,
                ]

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

        for index in range(count):

            name = str(
                feature_names[index]
            )

            value = float(
                values[index]
            )

            if "__" in name:
                name = name.split(
                    "__"
                )[-1]

            result.append(
                {
                    "feature": name,
                    "value": value,
                    "impact": value,
                    "abs_impact": abs(value),
                }
            )

        result.sort(
            key=lambda item:
                item["abs_impact"],
            reverse=True,
        )

        return {
            "available": True,
            "features": result[:15],
        }

    except Exception as e:

        print(
            "SHAP ERROR:",
            str(e),
        )

        return {
            "available": False,
            "features": [],
            "error": str(e),
        }


# ============================================================
# COUNTERFACTUAL
# ============================================================

def safe_predict(
    df: pd.DataFrame,
) -> Optional[int]:

    try:
        prediction = model.predict(
            df
        )

        return int(
            prediction[0]
        )

    except Exception as e:

        print(
            "Counterfactual prediction error:",
            e,
        )

        return None


def get_probabilities(
    df: pd.DataFrame,
) -> Dict[str, float]:

    try:

        probabilities = model.predict_proba(
            df
        )[0]

        return {
            "class_0_probability":
                float(probabilities[0]),

            "class_1_probability":
                float(probabilities[1]),
        }

    except Exception:

        return {
            "class_0_probability": 0.0,
            "class_1_probability": 0.0,
        }


def create_candidate_values(
    feature: str,
    current_value: Any,
) -> List[Any]:

    try:
        current = float(
            current_value
        )

    except Exception:
        return []

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
            value
            for value in values
            if abs(value - current) > 0.001
        ]

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
            float(value)
            for value in values
            if abs(value - current) > 0.001
        ]

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
            float(value)
            for value in values
            if abs(value - current) > 0.001
        ]

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
            float(value)
            for value in values
            if abs(value - current) > 0.001
        ]

    if feature == "age":

        values = [
            max(1, current - 10),
            max(1, current - 5),
            current + 5,
            min(100, current + 10),
        ]

        return [
            round(float(value), 2)
            for value in values
            if abs(value - current) > 0.001
        ]

    if feature in BINARY_FEATURES:

        if int(round(current)) == 0:
            return [1]

        return [0]

    return []


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

    search_features = [
        "TSH",
        "TT4",
        "T4U",
        "FTI",
        "age",
    ]

    try:

        shap_features = shap_result.get(
            "features",
            [],
        )

        for item in shap_features:

            feature = item.get(
                "feature"
            )

            if feature in FEATURE_NAMES:

                if feature not in search_features:

                    if feature not in MEASURED_FEATURES:

                        search_features.append(
                            feature
                        )

            if len(search_features) >= 8:
                break

    except Exception:
        pass

    search_features = search_features[:8]

    scenarios = []

    for feature in search_features:

        current_value = original_df.iloc[0][
            feature
        ]

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
                                "feature":
                                    feature,

                                "original_value":
                                    float(
                                        current_value
                                    ),

                                "counterfactual_value":
                                    float(
                                        new_value
                                    ),
                            }
                        ],

                        "prediction":
                            new_prediction,

                        "prediction_label":
                            (
                                "Thyroid Disease Predicted"
                                if new_prediction == 1
                                else
                                "Thyroid Disease Not Predicted"
                            ),

                        "class_0_probability":
                            probabilities[
                                "class_0_probability"
                            ],

                        "class_1_probability":
                            probabilities[
                                "class_1_probability"
                            ],

                        "distance":
                            distance,
                    }
                )

    scenarios.sort(
        key=lambda item:
            item["distance"]
    )

    unique = {}

    for scenario in scenarios:

        change = scenario["changes"][0]

        key = (
            change["feature"],
            change["counterfactual_value"],
        )

        if key not in unique:
            unique[key] = scenario

    scenarios = list(
        unique.values()
    )

    scenarios.sort(
        key=lambda item:
            item["distance"]
    )

    if not scenarios:

        return {
            "available": False,
            "features": [],
            "scenarios": [],
            "message": (
                "No counterfactual scenario "
                "was found within the tested "
                "feature ranges."
            ),
        }

    scenarios = scenarios[:3]

    return {
        "available": True,

        "features":
            scenarios[0]["changes"],

        "scenarios":
            scenarios,

        "target_prediction":
            scenarios[0]["prediction"],

        "target_label":
            scenarios[0]["prediction_label"],

        "message":
            "Counterfactual scenarios generated successfully.",
    }


# ============================================================
# DATASET PREPROCESSING
# ============================================================

def preprocess_dataset(
    dataframe: pd.DataFrame,
    target_column: Optional[str],
):

    df = dataframe.copy()

    original_rows = int(
        df.shape[0]
    )

    original_columns = int(
        df.shape[1]
    )

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    ).copy()

    empty_rows_removed = (
        original_rows
        - int(df.shape[0])
    )

    # Remove duplicates
    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # Detect target
    detected_target = detect_target_column(
        df
    )

    if (
        target_column is not None
        and target_column in df.columns
    ):
        final_target = target_column
    else:
        final_target = detected_target

    missing_before = int(
        df.isna().sum().sum()
    )

    missing_by_column = {
        str(column): int(value)
        for column, value
        in df.isna().sum().items()
        if int(value) > 0
    }

    feature_columns = [
        column
        for column in df.columns
        if column != final_target
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    # Fill numerical missing values
    for column in numeric_columns:

        if df[column].isna().any():

            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )

    # Fill categorical missing values
    for column in categorical_columns:

        if df[column].isna().any():

            mode = df[column].mode(
                dropna=True
            )

            if len(mode) > 0:
                replacement = mode.iloc[0]
            else:
                replacement = "Unknown"

            df[column] = df[column].fillna(
                replacement
            )

    # Remove rows with missing target
    target_missing_removed = 0

    if (
        final_target is not None
        and final_target in df.columns
    ):

        before_target = int(
            df.shape[0]
        )

        df = df.dropna(
            subset=[final_target]
        ).copy()

        target_missing_removed = (
            before_target
            - int(df.shape[0])
        )

    missing_after = int(
        df.isna().sum().sum()
    )

    # Target distribution
    target_distribution = {}

    if (
        final_target is not None
        and final_target in df.columns
    ):

        counts = df[final_target].value_counts(
            dropna=False
        )

        target_distribution = {
            str(key): int(value)
            for key, value
            in counts.items()
        }

    numeric_after = [
        str(column)
        for column
        in df.select_dtypes(
            include=np.number
        ).columns
        if column != final_target
    ]

    categorical_after = [
        str(column)
        for column in df.columns
        if (
            column != final_target
            and column not in numeric_after
        )
    ]

    # Save processed dataset
    df.to_csv(
        PREPROCESSED_DATASET_PATH,
        index=False,
    )

    info = {
        "success": True,

        "status": "preprocessed",

        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "processed_rows":
            int(df.shape[0]),

        "processed_columns":
            int(df.shape[1]),

        "empty_rows_removed":
            int(empty_rows_removed),

        "duplicate_rows_removed":
            int(duplicate_count),

        "target_missing_rows_removed":
            int(target_missing_removed),

        "missing_values_before":
            int(missing_before),

        "missing_values_after":
            int(missing_after),

        "missing_values_by_column":
            missing_by_column,

        "target_column":
            (
                str(final_target)
                if final_target is not None
                else None
            ),

        "target_distribution":
            target_distribution,

        "numeric_features":
            numeric_after,

        "categorical_features":
            categorical_after,

        "numeric_feature_count":
            len(numeric_after),

        "categorical_feature_count":
            len(categorical_after),

        "preprocessed_file":
            PREPROCESSED_DATASET_PATH.name,
    }

    return df, info


# ============================================================
# ADMIN TRAINING
# ============================================================

def make_preprocessor(
    X: pd.DataFrame,
):

    numeric_features = list(
        X.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_features = list(
        X.select_dtypes(
            exclude=np.number
        ).columns
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def calculate_roc_auc(
    model_pipeline,
    X_test,
    y_test,
    number_of_classes: int,
):

    try:

        if not hasattr(
            model_pipeline,
            "predict_proba",
        ):
            return None

        probabilities = (
            model_pipeline.predict_proba(
                X_test
            )
        )

        if number_of_classes == 2:

            return float(
                roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            )

        return float(
            roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )

    except Exception as e:

        print(
            "ROC-AUC warning:",
            str(e),
        )

        return None


def build_training_models(
    number_of_classes: int,
):

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=42,
                class_weight="balanced",
            ),
    }

    # Binary XGBoost
    if number_of_classes == 2:

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        )

    # Multiclass XGBoost
    else:

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=number_of_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
        )

    return models


def train_admin_models():

    global TRAINING_RESULTS
    global TRAINING_MODELS
    global TRAINING_INFO

    if PREPROCESSED_DATASET is None:

        raise ValueError(
            "Dataset must be preprocessed before training."
        )

    if not DATASET_TARGET:

        raise ValueError(
            "Target column could not be detected."
        )

    df = PREPROCESSED_DATASET.copy()

    target_column = DATASET_TARGET

    if target_column not in df.columns:

        target_column = detect_target_column(
            df
        )

    if not target_column:

        raise ValueError(
            "Target column was not found."
        )

    df = df.dropna(
        subset=[target_column]
    ).copy()

    if df.empty:

        raise ValueError(
            "Dataset contains no usable rows."
        )

    X = df.drop(
        columns=[target_column]
    )

    y_raw = df[target_column]

    if X.shape[1] == 0:

        raise ValueError(
            "Dataset contains no feature columns."
        )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw.astype(str)
    )

    class_count = len(
        label_encoder.classes_
    )

    if class_count < 2:

        raise ValueError(
            "Training requires at least 2 target classes."
        )

    if len(df) < 10:

        raise ValueError(
            "At least 10 rows are recommended for training."
        )

    test_size = 0.20

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        )

    except ValueError:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )
        )

    models = build_training_models(
        class_count
    )

    results = []
    trained_models = {}

    # --------------------------------------------------------
    # Train each algorithm
    # --------------------------------------------------------

    for model_name, classifier in models.items():

        print(
            f"Training {model_name}..."
        )

        try:

            preprocessor = make_preprocessor(
                X_train
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "classifier",
                        classifier,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            precision = precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )

            roc_auc = calculate_roc_auc(
                pipeline,
                X_test,
                y_test,
                class_count,
            )

            result = {

                "model":
                    model_name,

                "status":
                    "success",

                "accuracy":
                    round(
                        float(accuracy),
                        6,
                    ),

                "accuracy_percent":
                    round(
                        float(
                            accuracy * 100
                        ),
                        2,
                    ),

                "precision":
                    round(
                        float(precision),
                        6,
                    ),

                "recall":
                    round(
                        float(recall),
                        6,
                    ),

                "f1_score":
                    round(
                        float(f1),
                        6,
                    ),

                "roc_auc":
                    (
                        round(
                            float(roc_auc),
                            6,
                        )
                        if roc_auc is not None
                        else None
                    ),

                "test_samples":
                    int(len(y_test)),
            }

            results.append(
                result
            )

            trained_models[
                model_name
            ] = pipeline

            print(
                f"{model_name} completed: "
                f"accuracy={accuracy:.4f}"
            )

        except Exception as e:

            print(
                f"{model_name} ERROR:",
                str(e),
            )

            results.append(
                {
                    "model":
                        model_name,

                    "status":
                        "failed",

                    "accuracy":
                        None,

                    "accuracy_percent":
                        None,

                    "precision":
                        None,

                    "recall":
                        None,

                    "f1_score":
                        None,

                    "roc_auc":
                        None,

                    "test_samples":
                        int(len(y_test)),

                    "error":
                        str(e),
                }
            )

    # --------------------------------------------------------
    # Successful models
    # --------------------------------------------------------

    successful_results = [
        item
        for item in results
        if item["status"] == "success"
    ]

    successful_results.sort(
        key=lambda item:
            item["f1_score"],
        reverse=True,
    )

    # Keep original algorithm order
    model_order = {
        "Logistic Regression": 1,
        "Decision Tree": 2,
        "Random Forest": 3,
        "SVM": 4,
        "XGBoost": 5,
    }

    results.sort(
        key=lambda item:
            model_order.get(
                item["model"],
                99,
            )
    )

    TRAINING_RESULTS = results

    TRAINING_MODELS = trained_models

    TRAINING_INFO = {

        "dataset":
            DATASET_NAME,

        "target":
            target_column,

        "rows":
            int(df.shape[0]),

        "features":
            int(X.shape[1]),

        "training_rows":
            int(X_train.shape[0]),

        "testing_rows":
            int(X_test.shape[0]),

        "classes":
            [
                str(value)
                for value
                in label_encoder.classes_
            ],

        "class_count":
            int(class_count),

        "test_size":
            test_size,

        "random_state":
            42,

        "successful_models":
            len(successful_results),

        "total_models":
            len(results),
    }

    save_training_state()
    save_admin_state()

    return {

        "success":
            True,

        "message":
            "Model training completed.",

        "dataset":
            DATASET_NAME,

        "target":
            target_column,

        "training":
            TRAINING_INFO,

        "results":
            TRAINING_RESULTS,
    }


# ============================================================
# ADMIN SHAP / FEATURE IMPORTANCE
# ============================================================

def generate_admin_shap_summary():

    if not TRAINING_MODELS:

        return {

            "available":
                False,

            "models":
                [],

            "message":
                "No trained models are available.",
        }

    summary = []

    for model_name, trained_pipeline in (
        TRAINING_MODELS.items()
    ):

        try:

            classifier = (
                trained_pipeline.named_steps[
                    "classifier"
                ]
            )

            preprocessor = (
                trained_pipeline.named_steps[
                    "preprocessor"
                ]
            )

            feature_names = list(
                preprocessor.get_feature_names_out()
            )

            importance = None

            # Tree models
            if hasattr(
                classifier,
                "feature_importances_",
            ):

                importance = (
                    classifier.feature_importances_
                )

            # Linear models
            elif hasattr(
                classifier,
                "coef_",
            ):

                coefficients = (
                    classifier.coef_
                )

                if coefficients.ndim == 2:

                    importance = np.mean(
                        np.abs(coefficients),
                        axis=0,
                    )

                else:

                    importance = np.abs(
                        coefficients
                    )

            if importance is None:
                continue

            count = min(
                len(feature_names),
                len(importance),
            )

            features = []

            for index in range(count):

                name = str(
                    feature_names[index]
                )

                if "__" in name:

                    name = name.split(
                        "__"
                    )[-1]

                features.append(
                    {
                        "feature":
                            name,

                        "importance":
                            float(
                                importance[index]
                            ),
                    }
                )

            features.sort(
                key=lambda item:
                    item["importance"],
                reverse=True,
            )

            summary.append(
                {
                    "model":
                        model_name,

                    "features":
                        features[:15],
                }
            )

        except Exception as e:

            print(
                "ADMIN SHAP ERROR:",
                model_name,
                str(e),
            )

    return {

        "available":
            len(summary) > 0,

        "models":
            summary,
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    load_admin_state()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "ThyroCare AI API",

        "status":
            "online",

        "version":
            "7.1.0",

        "model_loaded":
            model is not None,

        "dataset_uploaded":
            DATASET is not None,

        "preprocessed":
            PREPROCESSED_DATASET is not None,

        "models_trained":
            len(TRAINING_RESULTS) > 0,

        "trained_model_count":
            len(TRAINING_MODELS),

        "prediction_history_count":
            len(PREDICTION_HISTORY),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            (
                "healthy"
                if model is not None
                else "unhealthy"
            ),

        "model_loaded":
            model is not None,

        "features":
            len(FEATURE_NAMES),

        "shap_available":
            True,

        "counterfactual_available":
            True,

        "dataset_uploaded":
            DATASET is not None,

        "dataset_preprocessed":
            PREPROCESSED_DATASET is not None,

        "models_trained":
            len(TRAINING_RESULTS) > 0,

        "trained_model_count":
            len(TRAINING_MODELS),

        "prediction_history_count":
            len(PREDICTION_HISTORY),

        "dataset_name":
            DATASET_NAME,

        "target":
            DATASET_TARGET,
    }


# ============================================================
# INFO
# ============================================================

@app.get("/info")
def info():

    return {

        "name":
            "ThyroCare AI",

        "model":
            "XGBoost",

        "features":
            FEATURE_NAMES,

        "feature_count":
            len(FEATURE_NAMES),

        "continuous_features":
            CONTINUOUS_FEATURES,

        "binary_features":
            BINARY_FEATURES,

        "explainability": {

            "shap":
                True,

            "counterfactual":
                True,
        },

        "dataset": {

            "uploaded":
                DATASET is not None,

            "name":
                DATASET_NAME,

            "target":
                DATASET_TARGET,

            "preprocessed":
                PREPROCESSED_DATASET is not None,
        },

        "training": {

            "trained":
                len(TRAINING_RESULTS) > 0,

            "models":
                [
                    item["model"]
                    for item
                    in TRAINING_RESULTS
                ],
        },

        "prediction_history": {

            "available":
                True,

            "count":
                len(PREDICTION_HISTORY),
        },
    }


# ============================================================
# ADMIN - UPLOAD DATASET
# ============================================================

@app.post("/admin/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
):

    global DATASET
    global DATASET_NAME
    global DATASET_TARGET

    global PREPROCESSED_DATASET
    global PREPROCESSING_INFO

    global TRAINING_RESULTS
    global TRAINING_MODELS
    global TRAINING_INFO

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )

    filename = file.filename.lower()

    if not filename.endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Save uploaded file
        with open(
            UPLOADED_DATASET_PATH,
            "wb",
        ) as output_file:

            output_file.write(
                contents
            )

        dataframe = pd.read_csv(
            UPLOADED_DATASET_PATH
        )

        if dataframe.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV contains no data."
                ),
            )

        if len(dataframe.columns) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV contains no columns."
                ),
            )

        target_column = detect_target_column(
            dataframe
        )

        DATASET = dataframe.copy()

        DATASET_NAME = file.filename

        DATASET_TARGET = target_column

        # Reset preprocessing
        PREPROCESSED_DATASET = None
        PREPROCESSING_INFO = None

        # Reset training
        TRAINING_RESULTS = []
        TRAINING_MODELS = {}
        TRAINING_INFO = None

        # Remove previous state files
        if PREPROCESSED_DATASET_PATH.exists():
            PREPROCESSED_DATASET_PATH.unlink()

        if TRAINING_RESULTS_PATH.exists():
            TRAINING_RESULTS_PATH.unlink()

        if TRAINING_MODELS_PATH.exists():
            TRAINING_MODELS_PATH.unlink()

        missing_values = int(
            dataframe.isna().sum().sum()
        )

        duplicate_rows = int(
            dataframe.duplicated().sum()
        )

        numeric_columns = int(
            dataframe.select_dtypes(
                include=np.number
            ).shape[1]
        )

        categorical_columns = int(
            len(dataframe.columns)
            - numeric_columns
        )

        save_admin_state()

        return {

            "success":
                True,

            "message":
                "Dataset uploaded successfully.",

            "filename":
                file.filename,

            "dataset_name":
                file.filename,

            "rows":
                int(dataframe.shape[0]),

            "columns":
                int(dataframe.shape[1]),

            "shape":
                [
                    int(dataframe.shape[0]),
                    int(dataframe.shape[1]),
                ],

            "column_names":
                [
                    str(column)
                    for column
                    in dataframe.columns
                ],

            "target_column":
                (
                    str(target_column)
                    if target_column is not None
                    else None
                ),

            "target":
                (
                    str(target_column)
                    if target_column is not None
                    else None
                ),

            "missing_values":
                missing_values,

            "duplicate_rows":
                duplicate_rows,

            "numeric_columns":
                numeric_columns,

            "categorical_columns":
                categorical_columns,

            "preprocessed":
                False,

            "models_trained":
                False,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "DATASET UPLOAD ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process dataset: "
                f"{str(e)}"
            ),
        )


# ============================================================
# ADMIN - DATASET STATUS
# ============================================================

@app.get("/admin/dataset")
def dataset_status():

    if DATASET is None:

        return {

            "uploaded":
                False,

            "message":
                "No dataset has been uploaded.",
        }

    return {

        "uploaded":
            True,

        "filename":
            DATASET_NAME,

        "rows":
            int(DATASET.shape[0]),

        "columns":
            int(DATASET.shape[1]),

        "shape":
            [
                int(DATASET.shape[0]),
                int(DATASET.shape[1]),
            ],

        "target_column":
            DATASET_TARGET,

        "column_names":
            [
                str(column)
                for column
                in DATASET.columns
            ],

        "missing_values":
            int(
                DATASET.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                DATASET.duplicated()
                .sum()
            ),

        "preprocessed":
            PREPROCESSED_DATASET is not None,

        "preprocessing":
            PREPROCESSING_INFO,

        "models_trained":
            len(TRAINING_RESULTS) > 0,

        "trained_model_count":
            len(TRAINING_MODELS),
    }


# ============================================================
# ADMIN - PREPROCESS
# ============================================================

@app.post("/admin/preprocess")
def admin_preprocess():

    global PREPROCESSED_DATASET
    global PREPROCESSING_INFO
    global DATASET_TARGET

    if DATASET is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No dataset has been uploaded. "
                "Upload a CSV dataset first."
            ),
        )

    try:

        processed_df, info = preprocess_dataset(
            DATASET,
            DATASET_TARGET,
        )

        PREPROCESSED_DATASET = (
            processed_df.copy()
        )

        PREPROCESSING_INFO = info

        detected_target = info.get(
            "target_column"
        )

        if detected_target:
            DATASET_TARGET = detected_target

        save_admin_state()

        return {

            "success":
                True,

            "message":
                (
                    "Dataset preprocessing "
                    "completed successfully."
                ),

            "dataset_name":
                DATASET_NAME,

            **info,
        }

    except Exception as e:

        print(
            "PREPROCESSING ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset preprocessing failed: "
                f"{str(e)}"
            ),
        )


# ============================================================
# ADMIN - PREPROCESS STATUS
# ============================================================

@app.get("/admin/preprocess")
def preprocessing_status():

    if PREPROCESSED_DATASET is None:

        return {

            "preprocessed":
                False,

            "message":
                "Dataset has not been preprocessed yet.",
        }

    return {

        "preprocessed":
            True,

        "dataset_name":
            DATASET_NAME,

        "information":
            PREPROCESSING_INFO,
    }


# ============================================================
# ADMIN - TRAIN
# ============================================================

@app.post("/admin/train")
def admin_train():

    if DATASET is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No dataset has been uploaded. "
                "Upload a CSV dataset first."
            ),
        )

    if PREPROCESSED_DATASET is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset has not been preprocessed. "
                "Click Preprocess Dataset first."
            ),
        )

    try:

        print(
            "============================================"
        )

        print(
            "Starting admin model training..."
        )

        result = train_admin_models()

        print(
            "Admin model training completed."
        )

        print(
            "============================================"
        )

        return result

    except Exception as e:

        print(
            "TRAINING ERROR:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Model training failed: "
                f"{str(e)}"
            ),
        )


# ============================================================
# ADMIN - MODEL COMPARISON
# ============================================================

@app.get("/admin/model-comparison")
def model_comparison():

    if not TRAINING_RESULTS:

        return {

            "trained":
                False,

            "message":
                "No models have been trained yet.",

            "results":
                [],
        }

    return {

        "trained":
            True,

        "dataset":
            DATASET_NAME,

        "target":
            DATASET_TARGET,

        "results":
            TRAINING_RESULTS,

        "training":
            TRAINING_INFO,
    }


# ============================================================
# ADMIN - METRICS
# ============================================================

@app.get("/admin/metrics")
def admin_metrics():

    if not TRAINING_RESULTS:

        return {

            "available":
                False,

            "message":
                "Training metrics are not available yet.",

            "metrics":
                [],
        }

    return {

        "available":
            True,

        "dataset":
            DATASET_NAME,

        "target":
            DATASET_TARGET,

        "metrics":
            TRAINING_RESULTS,

        "training":
            TRAINING_INFO,
    }


# ============================================================
# ADMIN - TRAINING STATUS
# ============================================================

@app.get("/admin/train")
def training_status():

    if not TRAINING_RESULTS:

        return {

            "trained":
                False,

            "message":
                "Models have not been trained yet.",
        }

    return {

        "trained":
            True,

        "training":
            TRAINING_INFO,

        "results":
            TRAINING_RESULTS,

        "trained_model_count":
            len(TRAINING_MODELS),
    }


# ============================================================
# ADMIN - SHAP / FEATURE IMPORTANCE
# ============================================================

@app.get("/admin/shap")
def admin_shap():

    return generate_admin_shap_summary()


# ============================================================
# ADMIN - PREDICTION HISTORY
# ============================================================

@app.get("/admin/prediction-history")
def prediction_history():

    return {

        "available":
            True,

        "count":
            len(PREDICTION_HISTORY),

        "history":
            PREDICTION_HISTORY,
    }


@app.delete("/admin/prediction-history")
def clear_prediction_history():

    global PREDICTION_HISTORY

    PREDICTION_HISTORY = []

    try:

        if PREDICTION_HISTORY_PATH.exists():
            PREDICTION_HISTORY_PATH.unlink()

    except Exception as e:

        print(
            "PREDICTION HISTORY CLEAR ERROR:",
            str(e),
        )

    return {

        "success":
            True,

        "message":
            "Prediction history cleared.",

        "count":
            0,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    req: PredictionRequest,
):

    global PREDICTION_HISTORY

    try:

        # ----------------------------------------------------
        # Convert request
        # ----------------------------------------------------

        df = request_to_dataframe(
            req
        )

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        result = get_prediction(
            df
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_result = generate_shap(
            df
        )

        # ----------------------------------------------------
        # Counterfactuals
        # ----------------------------------------------------

        counterfactual_result = (
            generate_counterfactuals(
                df,
                result["prediction"],
                shap_result,
            )
        )

        # ----------------------------------------------------
        # Save prediction history
        # ----------------------------------------------------

        next_id = 1

        if PREDICTION_HISTORY:

            existing_ids = []

            for item in PREDICTION_HISTORY:

                try:
                    existing_ids.append(
                        int(item.get("id", 0))
                    )
                except Exception:
                    pass

            if existing_ids:
                next_id = max(
                    existing_ids
                ) + 1

        history_record = {

            "id":
                next_id,

            "timestamp":
                datetime.now().isoformat(),

            "prediction":
                result["prediction"],

            "prediction_label":
                result["label"],

            "class_0_probability":
                result[
                    "class_0_probability"
                ],

            "class_1_probability":
                result[
                    "class_1_probability"
                ],

            "model":
                "XGBoost",
        }

        PREDICTION_HISTORY.insert(
            0,
            history_record,
        )

        # Keep latest 100 predictions
        PREDICTION_HISTORY = (
            PREDICTION_HISTORY[:100]
        )

        save_prediction_history()

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success":
                True,

            "prediction":
                result["prediction"],

            "prediction_label":
                result["label"],

            "label":
                result["label"],

            "model":
                "XGBoost",

            "class_0_probability":
                result[
                    "class_0_probability"
                ],

            "class_1_probability":
                result[
                    "class_1_probability"
                ],

            "probabilities": {

                "class_0":
                    result[
                        "class_0_probability"
                    ],

                "class_1":
                    result[
                        "class_1_probability"
                    ],
            },

            "shap_values":
                shap_result["features"],

            "shap":
                shap_result,

            "counterfactuals":
                counterfactual_result,

            "history_saved":
                True,

            "history_id":
                next_id,

            "prediction_history_count":
                len(PREDICTION_HISTORY),
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

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
)
