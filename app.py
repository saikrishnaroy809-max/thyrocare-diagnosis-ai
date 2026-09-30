# ============================================================
# THYROCARE AI - FASTAPI BACKEND
# Thyroid Disease Prediction + SHAP + Counterfactual AI
# Admin Dataset Upload + Dataset Preprocessing
# ============================================================

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="ThyroCare AI API",
    description=(
        "AI-powered thyroid disease prediction "
        "with Explainable AI and Admin Dataset Management"
    ),
    version="5.0.0",
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


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

model = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)

    print(
        "ThyroCare model loaded successfully."
    )

except Exception as e:

    MODEL_LOAD_ERROR = str(e)

    print(
        "MODEL LOAD ERROR:",
        MODEL_LOAD_ERROR,
    )


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
# ADMIN DATASET STATE
# ============================================================

DATASET = None
DATASET_NAME = None
DATASET_TARGET = None

PREPROCESSED_DATASET = None
PREPROCESSING_INFO = None


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

        "on thyroxine":
            req.on_thyroxine,

        "query on thyroxine":
            req.query_on_thyroxine,

        "on antithyroid medication":
            req.on_antithyroid_medication,

        "sick":
            req.sick,

        "pregnant":
            req.pregnant,

        "thyroid surgery":
            req.thyroid_surgery,

        "I131 treatment":
            req.I131_treatment,

        "query hypothyroid":
            req.query_hypothyroid,

        "query hyperthyroid":
            req.query_hyperthyroid,

        "lithium":
            req.lithium,

        "goitre":
            req.goitre,

        "tumor":
            req.tumor,

        "hypopituitary":
            req.hypopituitary,

        "psych":
            req.psych,

        "TSH measured":
            req.TSH_measured,

        "TSH":
            req.TSH,

        "T3 measured":
            req.T3_measured,

        "TT4 measured":
            req.TT4_measured,

        "TT4":
            req.TT4,

        "T4U measured":
            req.T4U_measured,

        "T4U":
            req.T4U,

        "FTI measured":
            req.FTI_measured,

        "FTI":
            req.FTI,
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
            "Model could not be loaded: "
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

        label = (
            "Thyroid Disease Predicted"
        )

    else:

        label = (
            "Thyroid Disease Not Predicted"
        )

    return {

        "prediction":
            prediction,

        "label":
            label,

        "class_0_probability":
            class_0_probability,

        "class_1_probability":
            class_1_probability,
    }


# ============================================================
# SHAP EXPLANATION
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

        if hasattr(
            model,
            "named_steps",
        ):

            if (
                "classifier"
                in model.named_steps
            ):

                classifier = (
                    model.named_steps[
                        "classifier"
                    ]
                )

            if (
                "preprocessor"
                in model.named_steps
            ):

                preprocessor = (
                    model.named_steps[
                        "preprocessor"
                    ]
                )

                transformed = (
                    preprocessor.transform(
                        df
                    )
                )

                try:

                    feature_names = list(
                        preprocessor
                        .get_feature_names_out()
                    )

                except Exception:

                    feature_names = (
                        FEATURE_NAMES.copy()
                    )

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = (
            explainer.shap_values(
                transformed
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

        values = values.astype(
            float
        )

        count = min(
            len(values),
            len(feature_names),
        )

        result = []

        for i in range(count):

            name = str(
                feature_names[i]
            )

            value = float(
                values[i]
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
            key=lambda x:
                x["abs_impact"],
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
# SAFE MODEL PREDICTION
# ============================================================

def safe_predict(
    df: pd.DataFrame,
) -> int | None:

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


# ============================================================
# SAFE PROBABILITIES
# ============================================================

def get_probabilities(
    df: pd.DataFrame,
) -> Dict[str, float]:

    try:

        probabilities = (
            model.predict_proba(df)[0]
        )

        return {

            "class_0_probability":
                float(
                    probabilities[0]
                ),

            "class_1_probability":
                float(
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
# ============================================================

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
            x
            for x in values
            if abs(x - current) > 0.001
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
            float(x)
            for x in values
            if abs(x - current) > 0.001
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
            float(x)
            for x in values
            if abs(x - current) > 0.001
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
            float(x)
            for x in values
            if abs(x - current) > 0.001
        ]

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

        old_value = (
            original.iloc[0][feature]
        )

        new_value = (
            modified.iloc[0][feature]
        )

        if old_value != new_value:

            total += feature_distance(
                feature,
                old_value,
                new_value,
            )

    return float(total)


# ============================================================
# COUNTERFACTUAL GENERATION
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
            []
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

        current_value = (
            original_df.iloc[0][feature]
        )

        candidate_values = (
            create_candidate_values(
                feature,
                current_value,
            )
        )

        for new_value in candidate_values:

            candidate_df = (
                original_df.copy()
            )

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

                probabilities = (
                    get_probabilities(
                        candidate_df
                    )
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
                                    (
                                        float(
                                            current_value
                                        )
                                        if isinstance(
                                            current_value,
                                            (
                                                np.integer,
                                                np.floating,
                                            ),
                                        )
                                        else
                                        current_value
                                    ),

                                "counterfactual_value":
                                    (
                                        float(
                                            new_value
                                        )
                                        if isinstance(
                                            new_value,
                                            (
                                                np.integer,
                                                np.floating,
                                            ),
                                        )
                                        else
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
        key=lambda x:
            x["distance"]
    )

    unique = {}

    for scenario in scenarios:

        change = (
            scenario["changes"][0]
        )

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
        key=lambda x:
            x["distance"]
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

    best_changes = (
        scenarios[0]["changes"]
    )

    return {

        "available":
            True,

        "features": [

            {
                "feature":
                    item["feature"],

                "original_value":
                    item["original_value"],

                "counterfactual_value":
                    item["counterfactual_value"],
            }

            for item in best_changes
        ],

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
    target_column: str | None,
):

    df = dataframe.copy()

    original_rows = int(
        df.shape[0]
    )

    original_columns = int(
        df.shape[1]
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    ).copy()

    empty_rows_removed = (
        original_rows -
        int(df.shape[0])
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # Detect target
    # --------------------------------------------------------

    detected_target = (
        detect_target_column(df)
    )

    if (
        target_column is not None
        and target_column in df.columns
    ):

        final_target = target_column

    else:

        final_target = detected_target

    # --------------------------------------------------------
    # Missing values before filling
    # --------------------------------------------------------

    missing_before = int(
        df.isna().sum().sum()
    )

    missing_by_column = {

        str(column):
            int(value)

        for column, value
        in df.isna().sum().items()

        if int(value) > 0
    }

    # --------------------------------------------------------
    # Separate target from features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fill numeric missing values
    # --------------------------------------------------------

    for column in numeric_columns:

        if df[column].isna().any():

            median_value = (
                df[column].median()
            )

            if pd.isna(median_value):

                median_value = 0

            df[column] = (
                df[column].fillna(
                    median_value
                )
            )

    # --------------------------------------------------------
    # Fill categorical missing values
    # --------------------------------------------------------

    for column in categorical_columns:

        if df[column].isna().any():

            mode = (
                df[column].mode(
                    dropna=True
                )
            )

            if len(mode) > 0:

                replacement = mode.iloc[0]

            else:

                replacement = "Unknown"

            df[column] = (
                df[column].fillna(
                    replacement
                )
            )

    # --------------------------------------------------------
    # Handle missing target values
    # --------------------------------------------------------

    target_missing_removed = 0

    if (
        final_target is not None
        and final_target in df.columns
    ):

        before_target = int(
            df.shape[0]
        )

        df = df.dropna(
            subset=[
                final_target
            ]
        ).copy()

        target_missing_removed = (
            before_target -
            int(df.shape[0])
        )

    # --------------------------------------------------------
    # Missing values after
    # --------------------------------------------------------

    missing_after = int(
        df.isna().sum().sum()
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_distribution = {}

    if (
        final_target is not None
        and final_target in df.columns
    ):

        counts = (
            df[final_target]
            .value_counts(
                dropna=False
            )
        )

        target_distribution = {

            str(key):
                int(value)

            for key, value
            in counts.items()
        }

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save preprocessed dataset
    # --------------------------------------------------------

    df.to_csv(
        PREPROCESSED_DATASET_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    info = {

        "success":
            True,

        "status":
            "preprocessed",

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
            "5.0.0",

        "model_loaded":
            model is not None,

        "dataset_uploaded":
            DATASET is not None,

        "preprocessed":
            PREPROCESSED_DATASET is not None,
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

        # ----------------------------------------------------
        # Save dataset
        # ----------------------------------------------------

        with open(
            UPLOADED_DATASET_PATH,
            "wb",
        ) as output_file:

            output_file.write(
                contents
            )

        # ----------------------------------------------------
        # Read dataset
        # ----------------------------------------------------

        dataframe = pd.read_csv(
            UPLOADED_DATASET_PATH
        )

        if dataframe.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV does not "
                    "contain any data rows."
                ),
            )

        if len(dataframe.columns) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded CSV does not "
                    "contain any columns."
                ),
            )

        # ----------------------------------------------------
        # Detect target
        # ----------------------------------------------------

        target_column = (
            detect_target_column(
                dataframe
            )
        )

        # ----------------------------------------------------
        # Store dataset
        # ----------------------------------------------------

        DATASET = dataframe.copy()

        DATASET_NAME = file.filename

        DATASET_TARGET = target_column

        # ----------------------------------------------------
        # Reset preprocessing
        # ----------------------------------------------------

        PREPROCESSED_DATASET = None

        PREPROCESSING_INFO = None

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        missing_values = int(
            dataframe.isna()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            dataframe.duplicated()
            .sum()
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

            "shape": [

                int(dataframe.shape[0]),

                int(dataframe.shape[1]),
            ],

            "column_names": [

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

        "shape": [

            int(DATASET.shape[0]),

            int(DATASET.shape[1]),
        ],

        "target_column":
            DATASET_TARGET,

        "column_names": [

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
    }


# ============================================================
# ADMIN - PREPROCESS DATASET
# ============================================================

@app.post("/admin/preprocess")
def admin_preprocess():

    global PREPROCESSED_DATASET
    global PREPROCESSING_INFO

    if DATASET is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No dataset has been uploaded. "
                "Upload a CSV dataset first."
            ),
        )

    try:

        print(
            "Starting dataset preprocessing..."
        )

        processed_df, info = (
            preprocess_dataset(
                DATASET,
                DATASET_TARGET,
            )
        )

        PREPROCESSED_DATASET = (
            processed_df.copy()
        )

        PREPROCESSING_INFO = info

        print(
            "Dataset preprocessing completed."
        )

        return {

            "success":
                True,

            "message":
                "Dataset preprocessing completed successfully.",

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
# ADMIN - PREPROCESSING STATUS
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
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    req: PredictionRequest,
):

    try:

        print(
            "Prediction request received."
        )

        # ----------------------------------------------------
        # Convert request
        # ----------------------------------------------------

        df = request_to_dataframe(
            req
        )

        print(
            "Input dataframe created."
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = get_prediction(
            df
        )

        print(
            "Prediction completed:",
            result["prediction"],
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_result = generate_shap(
            df
        )

        print(
            "SHAP completed."
        )

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

        print(
            "Counterfactual completed."
        )

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
                shap_result[
                    "features"
                ],

            "shap":
                shap_result,

            "counterfactuals":
                counterfactual_result,
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
