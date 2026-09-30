import React, { useRef, useState } from "react";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

export default function DatasetUpload() {
  const fileInputRef = useRef(null);

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

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("Please select a CSV file only.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
  };

  const openFilePicker = () => {
    setError("");
    setMessage("");

    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
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
          data?.missing_values ?? "—",

        duplicateRows:
          data?.duplicate_rows ?? "—",
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

  return (
    <section className="admin-feature-card dataset-upload-card">

      <div className="admin-feature-header">
        <div>
          <span className="admin-feature-icon">
            📤
          </span>

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

        {/* Native file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFileChange}
          className="dataset-native-input"
        />

        {/* File picker button */}
        <button
          type="button"
          className="dataset-file-button"
          onClick={openFilePicker}
        >
          📁 Choose CSV File
        </button>

        {/* Selected file */}
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
                {(file.size / 1024).toFixed(1)} KB
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

        {/* Upload button */}
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

        {/* Success */}
        {message && (
          <div className="dataset-success">
            ✅ {message}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="dataset-error">
            ❌ {error}
          </div>
        )}

      </div>

      {/* Dataset information */}
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

    </section>
  );
}

Important

This version does not use:

style={{ display: "none" }}

for the file input.

The actual file input remains available to the browser, while our button triggers it.

Now do only these steps

1. Replace "DatasetUpload.jsx".
2. Save it.
3. Commit it to GitHub.
4. Wait for Vercel → Ready.
5. Open the website again.
6. Go to Admin Dashboard.
7. Tap Choose CSV File.

You should now get the Android file picker.

Don't change "index.css" yet. First let's make sure the file picker actually opens.
