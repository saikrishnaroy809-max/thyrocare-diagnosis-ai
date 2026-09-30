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
    if (!datasetInfo) {
      setError(
        "Please upload the dataset before preprocessing."
      );
      return;
    }

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

      {/* ======================================================
          DATASET INFORMATION
          ====================================================== */}

      {datasetInfo && (
        <div className="dataset-info">

          <div className="dataset-info-title">
            📊 Dataset Information
          </div>

          <div className="dataset-info-grid">

            <div>
              <span>Dataset</span>
              <strong>
                {datasetInfo.filename}
              </strong>
            </div>

            <div>
              <span>Rows</span>
              <strong>
                {datasetInfo.rows}
              </strong>
            </div>

            <div>
              <span>Columns</span>
              <strong>
                {datasetInfo.columns}
              </strong>
            </div>

            <div>
              <span>Target</span>
              <strong>
                {datasetInfo.target}
              </strong>
            </div>

            <div>
              <span>Missing Values</span>
              <strong>
                {datasetInfo.missingValues}
              </strong>
            </div>

            <div>
              <span>Duplicate Rows</span>
              <strong>
                {datasetInfo.duplicateRows}
              </strong>
            </div>

          </div>

        </div>
      )}

      {/* ======================================================
          PREPROCESSING RESULTS
          ====================================================== */}

      {preprocessInfo && (
        <div className="dataset-info preprocessing-results">

          <div className="dataset-info-title">
            🧹 Preprocessing Results
          </div>

          <div className="dataset-info-grid">

            <div>
              <span>Original Rows</span>
              <strong>
                {preprocessInfo.original_rows ?? "—"}
              </strong>
            </div>

            <div>
              <span>Processed Rows</span>
              <strong>
                {preprocessInfo.processed_rows ?? "—"}
              </strong>
            </div>

            <div>
              <span>Original Columns</span>
              <strong>
                {preprocessInfo.original_columns ?? "—"}
              </strong>
            </div>

            <div>
              <span>Processed Columns</span>
              <strong>
                {preprocessInfo.processed_columns ?? "—"}
              </strong>
            </div>

            <div>
              <span>Duplicates Removed</span>
              <strong>
                {preprocessInfo.duplicate_rows_removed ?? "—"}
              </strong>
            </div>

            <div>
              <span>Empty Rows Removed</span>
              <strong>
                {preprocessInfo.empty_rows_removed ?? "—"}
              </strong>
            </div>

            <div>
              <span>Missing Before</span>
              <strong>
                {preprocessInfo.missing_values_before ?? "—"}
              </strong>
            </div>

            <div>
              <span>Missing After</span>
              <strong>
                {preprocessInfo.missing_values_after ?? "—"}
              </strong>
            </div>

            <div>
              <span>Target</span>
              <strong>
                {preprocessInfo.target_column ?? "—"}
              </strong>
            </div>

            <div>
              <span>Numeric Features</span>
              <strong>
                {preprocessInfo.numeric_feature_count ?? "—"}
              </strong>
            </div>

            <div>
              <span>Categorical Features</span>
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

          {/* NUMERIC FEATURES */}
          {preprocessInfo.numeric_features &&
            preprocessInfo.numeric_features.length > 0 && (

              <div className="preprocessing-feature-list">

                <h3>
                  🔢 Numeric Features
                </h3>

                <div
                  className="feature-tags"
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "8px",
                  }}
                >

                  {preprocessInfo.numeric_features.map(
                    (feature, index) => (

                      <span
                        key={`${feature}-${index}`}
                        style={{
                          display: "inline-block",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {feature}
                      </span>

                    )
                  )}

                </div>

              </div>

            )}

          {/* CATEGORICAL FEATURES */}
          {preprocessInfo.categorical_features &&
            preprocessInfo.categorical_features.length > 0 && (

              <div className="preprocessing-feature-list">

                <h3>
                  🔤 Categorical Features
                </h3>

                <div
                  className="feature-tags"
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "8px",
                  }}
                >

                  {preprocessInfo.categorical_features.map(
                    (feature, index) => (

                      <span
                        key={`${feature}-${index}`}
                        style={{
                          display: "inline-block",
                          whiteSpace: "nowrap",
                        }}
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

      {/* ======================================================
          TRAINING RESULTS
          ====================================================== */}

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
                {trainingInfo.dataset ??
                  trainingInfo.dataset_name ??
                  trainingInfo.filename ??
                  datasetInfo?.filename ??
                  "—"}
              </strong>
            </div>

            <div>
              <span>
                Target
              </span>

              <strong>
                {trainingInfo.target ??
                  trainingInfo.target_column ??
                  preprocessInfo?.target_column ??
                  "—"}
              </strong>
            </div>

            <div>
              <span>
                Models Trained
              </span>

              <strong>
                {trainingInfo.training?.successful_models ??
                  trainingInfo.results?.filter(
                    (model) =>
                      model.status === "success"
                  ).length ??
                  trainingInfo.results?.length ??
                  "—"}
                {" / "}
                {trainingInfo.training?.total_models ??
                  trainingInfo.results?.length ??
                  "—"}
              </strong>
            </div>

            <div>
              <span>
                Training Status
              </span>

              <strong>
                {trainingInfo.trained === true
                  ? "Completed"
                  : "Completed"}
              </strong>
            </div>

          </div>

          {/* TRAINING DETAILS */}
          {trainingInfo.training && (
            <div
              className="dataset-info-grid"
              style={{ marginTop: "16px" }}
            >

              <div>
                <span>
                  Total Rows
                </span>

                <strong>
                  {trainingInfo.training.rows ?? "—"}
                </strong>
              </div>

              <div>
                <span>
                  Features
                </span>

                <strong>
                  {trainingInfo.training.features ?? "—"}
                </strong>
              </div>

              <div>
                <span>
                  Training Rows
                </span>

                <strong>
                  {trainingInfo.training.training_rows ?? "—"}
                </strong>
              </div>

              <div>
                <span>
                  Testing Rows
                </span>

                <strong>
                  {trainingInfo.training.testing_rows ?? "—"}
                </strong>
              </div>

              <div>
                <span>
                  Classes
                </span>

                <strong>
                  {trainingInfo.training.class_count ?? "—"}
                </strong>
              </div>

              <div>
                <span>
                  Test Size
                </span>

                <strong>
                  {trainingInfo.training.test_size != null
                    ? `${(
                        trainingInfo.training.test_size * 100
                      ).toFixed(0)}%`
                    : "—"}
                </strong>
              </div>

            </div>
          )}

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
                        style={{
                          alignItems: "flex-start",
                          gap: "12px",
                          flexWrap: "wrap",
                        }}
                      >

                        <span>
                          {model.model ||
                            model.name ||
                            `Model ${index + 1}`}
                        </span>

                        <strong>
                          {model.accuracy_percent != null
                            ? `${model.accuracy_percent.toFixed(2)}%`
                            : typeof model.accuracy === "number"
                            ? `${(
                                model.accuracy * 100
                              ).toFixed(2)}%`
                            : "—"}
                        </strong>

                        <div
                          style={{
                            width: "100%",
                            display: "grid",
                            gridTemplateColumns:
                              "repeat(2, minmax(120px, 1fr))",
                            gap: "6px 12px",
                            marginTop: "6px",
                            fontSize: "13px",
                            opacity: 0.85,
                          }}
                        >

                          <span>
                            Precision:{" "}
                            {typeof model.precision === "number"
                              ? `${(
                                  model.precision * 100
                                ).toFixed(2)}%`
                              : "—"}
                          </span>

                          <span>
                            Recall:{" "}
                            {typeof model.recall === "number"
                              ? `${(
                                  model.recall * 100
                                ).toFixed(2)}%`
                              : "—"}
                          </span>

                          <span>
                            F1 Score:{" "}
                            {typeof model.f1_score === "number"
                              ? `${(
                                  model.f1_score * 100
                                ).toFixed(2)}%`
                              : "—"}
                          </span>

                          <span>
                            ROC-AUC:{" "}
                            {typeof model.roc_auc === "number"
                              ? model.roc_auc.toFixed(4)
                              : "—"}
                          </span>

                        </div>

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
