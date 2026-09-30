import React, { useRef, useState } from "react";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

export default function DatasetUpload() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [preprocessing, setPreprocessing] = useState(false);
  const [training, setTraining] = useState(false);

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [datasetInfo, setDatasetInfo] = useState(null);
  const [preprocessInfo, setPreprocessInfo] = useState(null);
  const [trainingInfo, setTrainingInfo] = useState(null);

  // ==========================================================
  // FILE SELECTION
  // ==========================================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    setMessage("");
    setError("");
    setDatasetInfo(null);
    setPreprocessInfo(null);
    setTrainingInfo(null);

    if (!selectedFile) {
      setFile(null);
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);

      setError("Please select a CSV file only.");

      event.target.value = "";
      return;
    }

    setFile(selectedFile);
  };

  // ==========================================================
  // OPEN FILE PICKER
  // ==========================================================

  const openFilePicker = () => {
    setError("");
    setMessage("");

    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // ==========================================================
  // UPLOAD DATASET
  // ==========================================================

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV dataset first.");
      return;
    }

    setLoading(true);
    setMessage("");
    setError("");
    setDatasetInfo(null);
    setPreprocessInfo(null);
    setTrainingInfo(null);

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/admin/upload-dataset`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Dataset upload failed."
        );
      }

      setMessage(
        data?.message ||
          "Dataset uploaded successfully."
      );

      setDatasetInfo({
        filename:
          data?.filename ||
          data?.dataset_name ||
          file.name,

        rows:
          data?.rows ??
          data?.shape?.[0] ??
          "—",

        columns:
          data?.columns ??
          data?.shape?.[1] ??
          "—",

        target:
          data?.target_column ||
          data?.target ||
          "Not detected",

        missingValues:
          data?.missing_values ??
          "—",

        duplicateRows:
          data?.duplicate_rows ??
          "—",
      });

    } catch (err) {
      console.error(
        "Dataset upload error:",
        err
      );

      setError(
        err?.message ||
          "Unable to upload dataset."
      );

    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // PREPROCESS DATASET
  // ==========================================================

  const handlePreprocess = async () => {
    setPreprocessing(true);
    setMessage("");
    setError("");
    setPreprocessInfo(null);
    setTrainingInfo(null);

    try {
      const response = await fetch(
        `${API_URL}/admin/preprocess`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Dataset preprocessing failed."
        );
      }

      setPreprocessInfo(data);

      setMessage(
        data?.message ||
          "Dataset preprocessing completed successfully."
      );

    } catch (err) {
      console.error(
        "Dataset preprocessing error:",
        err
      );

      setError(
        err?.message ||
          "Unable to preprocess dataset."
      );

    } finally {
      setPreprocessing(false);
    }
  };

  // ==========================================================
  // TRAIN MACHINE LEARNING ALGORITHMS
  // ==========================================================

  const handleTrain = async () => {
    if (!preprocessInfo) {
      setError(
        "Please preprocess the dataset before training."
      );
      return;
    }

    setTraining(true);
    setMessage("");
    setError("");
    setTrainingInfo(null);

    try {
      const response = await fetch(
        `${API_URL}/admin/train`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Model training failed."
        );
      }

      setTrainingInfo(data);

      setMessage(
        data?.message ||
          "All machine learning algorithms trained successfully."
      );

    } catch (err) {
      console.error(
        "Model training error:",
        err
      );

      setError(
        err?.message ||
          "Unable to train machine learning models."
      );

    } finally {
      setTraining(false);
    }
  };

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <section className="admin-feature-card dataset-upload-card">

      {/* HEADER */}
      <div className="admin-feature-header">

        <div>

          <span className="admin-feature-icon">
            📤
          </span>

          <h2>
            Upload Dataset
          </h2>

          <p>
            Upload a CSV dataset for
            preprocessing, model training
            and analysis.
          </p>

        </div>

      </div>


      {/* UPLOAD AREA */}
      <div className="dataset-upload-area">

        <div className="dataset-upload-icon">
          📁
        </div>

        <h3>
          Select your thyroid dataset
        </h3>

        <p>
          Only CSV files are supported.
        </p>


        {/* NATIVE FILE INPUT */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFileChange}
          className="dataset-native-input"
        />


        {/* FILE PICKER BUTTON */}
        <button
          type="button"
          className="dataset-file-button"
          onClick={openFilePicker}
        >
          📁 Choose CSV File
        </button>


        {/* SELECTED FILE */}
        {file && (

          <div className="selected-dataset">

            <span className="selected-dataset-icon">
              📄
            </span>

            <div className="selected-dataset-details">

              <strong>
                {file.name}
              </strong>

              <small>
                {(file.size / 1024).toFixed(1)}
                {" "}
                KB
              </small>

            </div>

            <button
              type="button"
              className="change-dataset-button"
              onClick={openFilePicker}
            >
              Change
            </button>

          </div>

        )}


        {/* UPLOAD BUTTON */}
        <button
          type="button"
          className="dataset-upload-button"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading
            ? "⏳ Uploading..."
            : "⬆️ Upload Dataset"}
        </button>


        {/* PREPROCESS BUTTON */}
        {datasetInfo && (

          <button
            type="button"
            className="dataset-preprocess-button"
            onClick={handlePreprocess}
            disabled={preprocessing}
          >
            {preprocessing
              ? "⚙️ Preprocessing..."
              : "🧹 Preprocess Dataset"}
          </button>

        )}


        {/* TRAIN BUTTON */}
        {preprocessInfo && (

          <button
            type="button"
            className="dataset-upload-button"
            onClick={handleTrain}
            disabled={training}
          >
            {training
              ? "⏳ Training Models..."
              : "🤖 Train Algorithms"}
          </button>

        )}


        {/* SUCCESS MESSAGE */}
        {message && (

          <div className="dataset-success">
            ✅ {message}
          </div>

        )}


        {/* ERROR MESSAGE */}
        {error && (

          <div className="dataset-error">
            ❌ {error}
          </div>

        )}

      </div>


      {/* DATASET INFORMATION */}
      {datasetInfo && (

        <div className="dataset-info">

          <div className="dataset-info-title">
            📊 Dataset Information
          </div>


          <div className="dataset-info-grid">

            <div>
              <span>
                Dataset
              </span>

              <strong>
                {datasetInfo.filename}
              </strong>
            </div>


            <div>
              <span>
                Rows
              </span>

              <strong>
                {datasetInfo.rows}
              </strong>
            </div>


            <div>
              <span>
                Columns
              </span>

              <strong>
                {datasetInfo.columns}
              </strong>
            </div>


            <div>
              <span>
                Target
              </span>

              <strong>
                {datasetInfo.target}
              </strong>
            </div>


            <div>
              <span>
                Missing Values
              </span>

              <strong>
                {datasetInfo.missingValues}
              </strong>
            </div>


            <div>
              <span>
                Duplicate Rows
              </span>

              <strong>
                {datasetInfo.duplicateRows}
              </strong>
            </div>

          </div>

        </div>

      )}


      {/* PREPROCESSING RESULTS */}
      {preprocessInfo && (

        <div className="dataset-info preprocessing-results">

          <div className="dataset-info-title">
            🧹 Preprocessing Results
          </div>


          <div className="dataset-info-grid">

            <div>
              <span>
                Original Rows
              </span>

              <strong>
                {preprocessInfo.original_rows ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Processed Rows
              </span>

              <strong>
                {preprocessInfo.processed_rows ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Original Columns
              </span>

              <strong>
                {preprocessInfo.original_columns ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Processed Columns
              </span>

              <strong>
                {preprocessInfo.processed_columns ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Duplicates Removed
              </span>

              <strong>
                {preprocessInfo.duplicate_rows_removed ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Empty Rows Removed
              </span>

              <strong>
                {preprocessInfo.empty_rows_removed ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Missing Before
              </span>

              <strong>
                {preprocessInfo.missing_values_before ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Missing After
              </span>

              <strong>
                {preprocessInfo.missing_values_after ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Target
              </span>

              <strong>
                {preprocessInfo.target_column ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Numeric Features
              </span>

              <strong>
                {preprocessInfo.numeric_feature_count ?? "—"}
              </strong>
            </div>


            <div>
              <span>
                Categorical Features
              </span>

              <strong>
                {preprocessInfo.categorical_feature_count ?? "—"}
              </strong>
            </div>

          </div>


          {/* CLASS DISTRIBUTION */}
          {preprocessInfo.target_distribution &&
            Object.keys(
              preprocessInfo.target_distribution
            ).length > 0 && (

              <div className="preprocessing-distribution">

                <h3>
                  🎯 Target Class Distribution
                </h3>

                <div className="distribution-list">

                  {Object.entries(
                    preprocessInfo.target_distribution
                  ).map(
                    ([key, value]) => (

                      <div
                        className="distribution-item"
                        key={key}
                      >

                        <span>
                          Class {key}
                        </span>

                        <strong>
                          {value}
                        </strong>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}


          {/* FEATURE INFORMATION */}
          {preprocessInfo.numeric_features &&
            preprocessInfo.numeric_features.length > 0 && (

              <div className="preprocessing-feature-list">

                <h3>
                  🔢 Numeric Features
                </h3>

                <div className="feature-tags">

                  {preprocessInfo.numeric_features.map(
                    (feature) => (

                      <span
                        key={feature}
                      >
                        {feature}
                      </span>

                    )
                  )}

                </div>

              </div>

            )}


          {preprocessInfo.categorical_features &&
            preprocessInfo.categorical_features.length > 0 && (

              <div className="preprocessing-feature-list">

                <h3>
                  🔤 Categorical Features
                </h3>

                <div className="feature-tags">

                  {preprocessInfo.categorical_features.map(
                    (feature) => (

                      <span
                        key={feature}
                      >
                        {feature}
                      </span>

                    )
                  )}

                </div>

              </div>

            )}

        </div>

      )}


      {/* TRAINING RESULTS */}
      {trainingInfo && (

        <div className="dataset-info training-results">

          <div className="dataset-info-title">
            🤖 Model Training Results
          </div>


          {/* TRAINING SUMMARY */}
          <div className="dataset-info-grid">

            <div>
              <span>
                Dataset
              </span>

              <strong>
                {trainingInfo.dataset_name ??
                  trainingInfo.filename ??
                  "—"}
              </strong>
            </div>


            <div>
              <span>
                Target
              </span>

              <strong>
                {trainingInfo.target_column ??
                  trainingInfo.target ??
                  "—"}
              </strong>
            </div>


            <div>
              <span>
                Models Trained
              </span>

              <strong>
                {trainingInfo.results?.length ??
                  trainingInfo.models_trained?.length ??
                  "—"}
              </strong>
            </div>


            <div>
              <span>
                Training Status
              </span>

              <strong>
                Completed
              </strong>
            </div>

          </div>


          {/* MODEL RESULTS */}
          {trainingInfo.results &&
            Array.isArray(trainingInfo.results) &&
            trainingInfo.results.length > 0 && (

              <div className="preprocessing-distribution">

                <h3>
                  📈 Algorithm Performance
                </h3>


                <div className="distribution-list">

                  {trainingInfo.results.map(
                    (model, index) => (

                      <div
                        className="distribution-item"
                        key={
                          model.model ||
                          model.name ||
                          index
                        }
                      >

                        <span>
                          {model.model ||
                            model.name ||
                            `Model ${index + 1}`}
                        </span>

                        <strong>
                          {typeof model.accuracy === "number"
                            ? `${(
                                model.accuracy * 100
                              ).toFixed(2)}%`
                            : model.accuracy ?? "—"}
                        </strong>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}

        </div>

      )}

    </section>
  );
}

What changed?

Only one new workflow was added:

Upload Dataset
↓
Preprocess Dataset
↓
🤖 Train Algorithms
↓
Backend "/admin/train"
↓
5 ML algorithms are trained

The training button appears only after preprocessing succeeds, so the user cannot accidentally train before preprocessing.

Your current workflow

After replacing the file:

1. Save "DatasetUpload.jsx".
2. Push the change to GitHub.
3. Vercel will deploy automatically.
4. Open your Admin Dashboard.
5. Upload "new-thyroid.csv".
6. Click Upload Dataset.
7. Click Preprocess Dataset.
8. You should then see 🤖 Train Algorithms.
9. Click it.

Don't change "index.css" yet. Your existing CSS will handle the button using the existing ".dataset-upload-button" styling.

After you click Train Algorithms, send me the result/error you see. Then we'll add the Model Comparison graph + detailed accuracy/precision/recall/F1/ROC-AUC section without redesigning your existing dashboard.
