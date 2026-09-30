import React, { useState } from "react";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

export default function DatasetUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [datasetInfo, setDatasetInfo] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    setMessage("");
    setError("");
    setDatasetInfo(null);

    if (!selectedFile) {
      setFile(null);
      return;
    }

    const isCSV =
      selectedFile.name.toLowerCase().endsWith(".csv");

    if (!isCSV) {
      setFile(null);
      setError("Please select a CSV file.");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV dataset first.");
      return;
    }

    setLoading(true);
    setMessage("");
    setError("");
    setDatasetInfo(null);

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
          data?.row_count ??
          "—",

        columns:
          data?.columns ??
          data?.shape?.[1] ??
          data?.column_count ??
          "—",

        target:
          data?.target_column ||
          data?.target ||
          "Not detected",
      });
    } catch (err) {
      setError(
        err?.message ||
          "Unable to upload dataset."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="admin-feature-card dataset-upload-card">
      <div className="admin-feature-header">
        <div>
          <span className="admin-feature-icon">📤</span>

          <h2>Upload Dataset</h2>

          <p>
            Upload a CSV dataset for preprocessing,
            model training and analysis.
          </p>
        </div>
      </div>

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

        <label
          className="dataset-file-button"
          htmlFor="thyroid-dataset-file"
        >
          Choose CSV File
        </label>

        <input
          id="thyroid-dataset-file"
          type="file"
          accept=".csv,text/csv"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {file && (
          <div className="selected-dataset">
            <span>📄</span>

            <div>
              <strong>{file.name}</strong>

              <small>
                {(file.size / 1024).toFixed(1)} KB
              </small>
            </div>
          </div>
        )}

        <button
          type="button"
          className="dataset-upload-button"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading
            ? "Uploading..."
            : "Upload Dataset"}
        </button>

        {message && (
          <div className="dataset-success">
            ✅ {message}
          </div>
        )}

        {error && (
          <div className="dataset-error">
            ❌ {error}
          </div>
        )}
      </div>

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
          </div>
        </div>
      )}
    </section>
  );
              }
