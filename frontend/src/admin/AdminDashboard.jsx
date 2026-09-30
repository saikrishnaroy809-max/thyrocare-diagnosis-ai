import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import DatasetUpload from "./DatasetUpload";

/* =========================================================
   THYROCARE AI
   ADMIN DASHBOARD
========================================================= */

const API_URL =
  "https://thyrocare-diagnosis-ai.onrender.com";

export default function AdminDashboard({
  result,
  analysisHistory = [],
  onNewAnalysis,
  onViewResults,
  onLogout,
}) {
  /* =========================================================
     MODEL COMPARISON STATE
  ========================================================= */

  const [modelComparison, setModelComparison] =
    useState([]);

  const [comparisonLoading, setComparisonLoading] =
    useState(false);

  const [comparisonError, setComparisonError] =
    useState("");

  /* =========================================================
     PERSISTENT PREDICTION HISTORY STATE
  ========================================================= */

  const [predictionHistory, setPredictionHistory] =
    useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  const [historyError, setHistoryError] =
    useState("");

  /* =========================================================
     BASIC RESULT DATA
  ========================================================= */

  const prediction =
    result?.prediction ??
    result?.predicted_class ??
    result?.class ??
    null;

  const modelName =
    result?.model_name ??
    result?.model ??
    result?.algorithm ??
    "XGBoost";

  const probabilities =
    result?.probabilities ??
    result?.class_probabilities ??
    {};

  const class0Probability =
    probabilities?.["0"] ??
    probabilities?.class_0 ??
    probabilities?.not_disease ??
    result?.class_0_probability ??
    0;

  const class1Probability =
    probabilities?.["1"] ??
    probabilities?.class_1 ??
    probabilities?.disease ??
    result?.class_1_probability ??
    0;

  /* =========================================================
     FETCH MODEL COMPARISON
  ========================================================= */

  const loadModelComparison = async () => {
    setComparisonLoading(true);
    setComparisonError("");

    try {
      const response = await fetch(
        `${API_URL}/admin/model-comparison`
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      if (
        data?.trained &&
        Array.isArray(data?.results)
      ) {
        setModelComparison(data.results);
      } else {
        setModelComparison([]);
      }
    } catch (error) {
      console.error(
        "Model comparison error:",
        error
      );

      setComparisonError(
        "Unable to load model comparison results."
      );
    } finally {
      setComparisonLoading(false);
    }
  };

  /* =========================================================
     FETCH PERSISTENT PREDICTION HISTORY
  ========================================================= */

  const loadPredictionHistory = async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(
        `${API_URL}/admin/prediction-history`
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      if (
        data?.available &&
        Array.isArray(data?.history)
      ) {
        setPredictionHistory(data.history);
      } else {
        setPredictionHistory([]);
      }
    } catch (error) {
      console.error(
        "Prediction history error:",
        error
      );

      setHistoryError(
        "Unable to load prediction history."
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  /* =========================================================
     INITIAL LOAD
  ========================================================= */

  useEffect(() => {
    loadModelComparison();
    loadPredictionHistory();
  }, []);

  /* =========================================================
     HELPERS
  ========================================================= */

  const formatPercent = (value) => {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "0.00%";
    }

    const percentage =
      number <= 1
        ? number * 100
        : number;

    return `${percentage.toFixed(2)}%`;
  };

  const formatMetric = (value) => {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "0.00%";
    }

    const percentage =
      number <= 1
        ? number * 100
        : number;

    return `${percentage.toFixed(2)}%`;
  };

  const formatDate = (value) => {
    if (!value) {
      return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
      return String(value);
    }

    return date.toLocaleString();
  };

  /* =========================================================
     CHART DATA
  ========================================================= */

  const chartData = useMemo(() => {
    return modelComparison
      .filter(
        (item) =>
          item?.status === "success" ||
          item?.status === undefined
      )
      .map((item) => ({
        name: item?.model ?? "Model",

        Accuracy: Number(
          item?.accuracy_percent ??
            Number(item?.accuracy ?? 0) * 100
        ),

        Precision: Number(
          item?.precision_percent ??
            Number(item?.precision ?? 0) * 100
        ),

        Recall: Number(
          item?.recall_percent ??
            Number(item?.recall ?? 0) * 100
        ),

        F1: Number(
          item?.f1_percent ??
            Number(item?.f1_score ?? 0) * 100
        ),
      }));
  }, [modelComparison]);

  /* =========================================================
     BEST DISPLAY METRICS
  ========================================================= */

  const comparisonSummary = useMemo(() => {
    if (!modelComparison.length) {
      return {
        models: 0,
        averageAccuracy: 0,
      };
    }

    const valid = modelComparison.filter(
      (item) =>
        item?.status === "success" ||
        item?.status === undefined
    );

    if (!valid.length) {
      return {
        models: 0,
        averageAccuracy: 0,
      };
    }

    const average =
      valid.reduce(
        (sum, item) =>
          sum +
          Number(
            item?.accuracy_percent ??
              Number(item?.accuracy ?? 0) * 100
          ),
        0
      ) / valid.length;

    return {
      models: valid.length,
      averageAccuracy: average,
    };
  }, [modelComparison]);

  /* =========================================================
     PREDICTION LABEL
  ========================================================= */

  const predictionLabel =
    Number(prediction) === 1
      ? "Thyroid Disease Predicted"
      : Number(prediction) === 0
      ? "Thyroid Disease Not Predicted"
      : "No prediction yet";

  const predictionClass =
    Number(prediction) === 1
      ? "positive"
      : Number(prediction) === 0
      ? "negative"
      : "neutral";

  /* =========================================================
     SHAP DATA
  ========================================================= */

  const shapValues =
    result?.shap_values ??
    result?.shap ??
    result?.feature_importance ??
    [];

  const topShap = useMemo(() => {
    let values = shapValues;

    if (
      !Array.isArray(values) &&
      values &&
      Array.isArray(values.features)
    ) {
      values = values.features;
    }

    if (!Array.isArray(values)) {
      return [];
    }

    return values
      .map((item, index) => {
        if (typeof item === "number") {
          return {
            feature: `Feature ${index + 1}`,
            value: Number(item),
          };
        }

        const value = Number(
          item?.shap_value ??
            item?.value ??
            item?.impact ??
            0
        );

        return {
          feature:
            item?.feature ??
            item?.name ??
            item?.feature_name ??
            `Feature ${index + 1}`,

          value:
            Number.isFinite(value)
              ? value
              : 0,
        };
      })
      .sort(
        (a, b) =>
          Math.abs(b.value) -
          Math.abs(a.value)
      )
      .slice(0, 5);
  }, [shapValues]);

  /* =========================================================
     HISTORY COUNT
  ========================================================= */

  const historyCount =
    predictionHistory.length;

  /* =========================================================
     VIEW RESULTS
  ========================================================= */

  const handleViewResults = () => {
    if (!result) {
      return;
    }

    if (
      typeof onViewResults ===
      "function"
    ) {
      onViewResults();
    }
  };

  /* =========================================================
     NEW ANALYSIS
  ========================================================= */

  const handleNewAnalysis = () => {
    if (
      typeof onNewAnalysis ===
      "function"
    ) {
      onNewAnalysis();
    }
  };

  /* =========================================================
     LOGOUT
  ========================================================= */

  const handleLogout = () => {
    if (
      typeof onLogout ===
      "function"
    ) {
      onLogout();
    }
  };

  /* =========================================================
     RENDER
  ========================================================= */

  return (
    <main className="admin-dashboard">

      <div className="admin-dashboard-container">

        {/* =====================================================
            TOP BAR
        ===================================================== */}

        <section className="admin-topbar">

          <div className="admin-topbar-brand">

            <div className="admin-brand-mark">
              TC
            </div>

            <div>

              <span className="admin-eyebrow">
                THYROCARE AI
              </span>

              <h1>
                Admin Dashboard
              </h1>

              <p>
                Monitor prediction activity,
                model responses and explainability.
              </p>

            </div>

          </div>

          <div className="admin-topbar-actions">

            <button
              type="button"
              className="admin-secondary-button"
              onClick={handleNewAnalysis}
            >
              <span>＋</span>
              New Analysis
            </button>

            <button
              type="button"
              className="admin-secondary-button"
              onClick={handleViewResults}
              disabled={!result}
            >
              View Results
            </button>

            <button
              type="button"
              className="admin-secondary-button admin-danger-button"
              onClick={handleLogout}
            >
              Logout
            </button>

          </div>

        </section>

        {/* =====================================================
            STATISTICS
        ===================================================== */}

        <section className="admin-stats-grid">

          <div className="admin-stat-card">

            <div className="admin-stat-icon">
              📊
            </div>

            <div className="admin-stat-content">

              <span className="admin-stat-label">
                Total Analyses
              </span>

              <strong className="admin-stat-value">
                {historyCount}
              </strong>

              <span className="admin-stat-subtext">
                Backend prediction records
              </span>

            </div>

          </div>

          <div className="admin-stat-card">

            <div className="admin-stat-icon">
              🤖
            </div>

            <div className="admin-stat-content">

              <span className="admin-stat-label">
                Current Model
              </span>

              <strong className="admin-stat-value admin-model-value">
                {modelName}
              </strong>

              <span className="admin-stat-subtext">
                Machine learning model
              </span>

            </div>

          </div>

          <div className="admin-stat-card">

            <div className="admin-stat-icon">
              🧠
            </div>

            <div className="admin-stat-content">

              <span className="admin-stat-label">
                Latest Class
              </span>

              <strong className="admin-stat-value">
                {prediction === null
                  ? "—"
                  : `Class ${prediction}`}
              </strong>

              <span className="admin-stat-subtext">
                Latest prediction
              </span>

            </div>

          </div>

          <div className="admin-stat-card">

            <div className="admin-stat-icon">
              ⚡
            </div>

            <div className="admin-stat-content">

              <span className="admin-stat-label">
                System
              </span>

              <strong className="admin-stat-value">
                Ready
              </strong>

              <span className="admin-stat-subtext">
                Prediction interface available
              </span>

            </div>

          </div>

        </section>

        {/* =====================================================
            DATASET UPLOAD
        ===================================================== */}

        <section className="admin-feature-section">

          <DatasetUpload />

        </section>

        {/* =====================================================
            MODEL COMPARISON
        ===================================================== */}

        <section
          className="admin-panel admin-model-comparison-panel"
          style={{
            marginTop: "20px",
          }}
        >

          <div className="admin-panel-header">

            <div>

              <span className="admin-panel-badge">
                COMPARE MODELS
              </span>

              <h2>
                Model Comparison
              </h2>

              <p>
                Compare the performance of all
                trained machine-learning algorithms.
              </p>

            </div>

            <button
              type="button"
              className="admin-secondary-button"
              onClick={loadModelComparison}
              disabled={comparisonLoading}
            >
              {comparisonLoading
                ? "Refreshing..."
                : "↻ Refresh"}
            </button>

          </div>

          {comparisonLoading ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                📈
              </div>

              <h3>
                Loading Model Results
              </h3>

              <p>
                Fetching trained model performance
                from the prediction server.
              </p>

            </div>

          ) : comparisonError ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                ⚠️
              </div>

              <h3>
                Comparison Unavailable
              </h3>

              <p>
                {comparisonError}
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={loadModelComparison}
              >
                Try Again
              </button>

            </div>

          ) : chartData.length === 0 ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                📊
              </div>

              <h3>
                No Model Results Yet
              </h3>

              <p>
                Upload, preprocess and train a
                dataset to display model comparison.
              </p>

            </div>

          ) : (

            <>

              {/* =================================================
                  SUMMARY CARDS
              ================================================= */}

              <div
                className="admin-stats-grid"
                style={{
                  marginTop: "20px",
                }}
              >

                <div className="admin-stat-card">

                  <div className="admin-stat-icon">
                    🤖
                  </div>

                  <div className="admin-stat-content">

                    <span className="admin-stat-label">
                      Models Trained
                    </span>

                    <strong className="admin-stat-value">
                      {comparisonSummary.models}
                    </strong>

                    <span className="admin-stat-subtext">
                      Successful algorithms
                    </span>

                  </div>

                </div>

                <div className="admin-stat-card">

                  <div className="admin-stat-icon">
                    🎯
                  </div>

                  <div className="admin-stat-content">

                    <span className="admin-stat-label">
                      Average Accuracy
                    </span>

                    <strong className="admin-stat-value">
                      {comparisonSummary.averageAccuracy.toFixed(
                        2
                      )}
                      %
                    </strong>

                    <span className="admin-stat-subtext">
                      Across trained models
                    </span>

                  </div>

                </div>

              </div>

              {/* =================================================
                  CHART
              ================================================= */}

              <div
                style={{
                  width: "100%",
                  height: 430,
                  marginTop: "28px",
                }}
              >

                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >

                  <BarChart
                    data={chartData}
                    margin={{
                      top: 20,
                      right: 20,
                      left: 0,
                      bottom: 70,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                      opacity={0.18}
                    />

                    <XAxis
                      dataKey="name"
                      angle={-18}
                      textAnchor="end"
                      interval={0}
                      height={80}
                      tick={{
                        fill: "currentColor",
                        fontSize: 12,
                      }}
                    />

                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(value) =>
                        `${value}%`
                      }
                      tick={{
                        fill: "currentColor",
                        fontSize: 12,
                      }}
                    />

                    <Tooltip
                      formatter={(value) =>
                        `${Number(value).toFixed(2)}%`
                      }
                    />

                    <Legend />

                    <Bar
                      dataKey="Accuracy"
                      name="Accuracy"
                      fill="#25c7f5"
                      radius={[5, 5, 0, 0]}
                    />

                    <Bar
                      dataKey="Precision"
                      name="Precision"
                      fill="#7c5cff"
                      radius={[5, 5, 0, 0]}
                    />

                    <Bar
                      dataKey="Recall"
                      name="Recall"
                      fill="#27d7a0"
                      radius={[5, 5, 0, 0]}
                    />

                    <Bar
                      dataKey="F1"
                      name="F1 Score"
                      fill="#ffb84d"
                      radius={[5, 5, 0, 0]}
                    />

                  </BarChart>

                </ResponsiveContainer>

              </div>

              {/* =================================================
                  METRICS TABLE
              ================================================= */}

              <div
                className="admin-table-wrapper"
                style={{
                  marginTop: "25px",
                }}
              >

                <table className="admin-table">

                  <thead>

                    <tr>
                      <th>Model</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>ROC-AUC</th>
                    </tr>

                  </thead>

                  <tbody>

                    {modelComparison.map(
                      (item, index) => (

                        <tr
                          key={
                            item?.model ??
                            `model-${index}`
                          }
                        >

                          <td>
                            <span className="admin-model-tag">
                              {item?.model ??
                                "Model"}
                            </span>
                          </td>

                          <td>
                            {formatMetric(
                              item?.accuracy_percent ??
                                item?.accuracy ??
                                0
                            )}
                          </td>

                          <td>
                            {formatMetric(
                              item?.precision ??
                                0
                            )}
                          </td>

                          <td>
                            {formatMetric(
                              item?.recall ??
                                0
                            )}
                          </td>

                          <td>
                            {formatMetric(
                              item?.f1_score ??
                                0
                            )}
                          </td>

                          <td>
                            {Number(
                              item?.roc_auc ?? 0
                            ).toFixed(4)}
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>

            </>

          )}

        </section>

        {/* =====================================================
            MAIN GRID
        ===================================================== */}

        <section className="admin-content-grid">

          {/* ===================================================
              LATEST ANALYSIS
          =================================================== */}

          <div className="admin-panel">

            <div className="admin-panel-header">

              <div>

                <span className="admin-panel-badge">
                  LATEST
                </span>

                <h2>
                  Latest Analysis
                </h2>

                <p>
                  Most recent model prediction
                  generated by ThyroCare AI.
                </p>

              </div>

              <span className="admin-status">

                <span className="admin-status-dot" />

                System Ready

              </span>

            </div>

            {!result ? (

              <div className="admin-empty">

                <div className="admin-empty-icon">
                  🔍
                </div>

                <h3>
                  No Analysis Yet
                </h3>

                <p>
                  Run a prediction to display
                  the latest model response here.
                </p>

                <button
                  type="button"
                  className="primary-button"
                  onClick={handleNewAnalysis}
                >
                  Start Analysis
                </button>

              </div>

            ) : (

              <div className="admin-prediction-content">

                <div
                  className={`admin-prediction-box ${predictionClass}`}
                >

                  <span className="prediction-label">
                    MODEL PREDICTION
                  </span>

                  <h3>
                    {predictionLabel}
                  </h3>

                  <div className="admin-prediction-class">
                    Class{" "}
                    {prediction === null
                      ? "—"
                      : prediction}
                  </div>

                </div>

                <div className="admin-probabilities">

                  <div className="admin-probability">

                    <div className="admin-probability-header">

                      <span>
                        Class 0
                      </span>

                      <strong>
                        {formatPercent(
                          class0Probability
                        )}
                      </strong>

                    </div>

                    <div className="admin-progress">

                      <div
                        className="admin-progress-bar"
                        style={{
                          width: `${Math.min(
                            Number(
                              class0Probability <= 1
                                ? class0Probability * 100
                                : class0Probability
                            ) || 0,
                            100
                          )}%`,
                        }}
                      />

                    </div>

                    <small>
                      Thyroid disease not predicted
                    </small>

                  </div>

                  <div className="admin-probability">

                    <div className="admin-probability-header">

                      <span>
                        Class 1
                      </span>

                      <strong>
                        {formatPercent(
                          class1Probability
                        )}
                      </strong>

                    </div>

                    <div className="admin-progress">

                      <div
                        className="admin-progress-bar"
                        style={{
                          width: `${Math.min(
                            Number(
                              class1Probability <= 1
                                ? class1Probability * 100
                                : class1Probability
                            ) || 0,
                            100
                          )}%`,
                        }}
                      />

                    </div>

                    <small>
                      Thyroid disease predicted
                    </small>

                  </div>

                </div>

                <button
                  type="button"
                  className="primary-button admin-view-results-button"
                  onClick={handleViewResults}
                >
                  Open Full Results →
                </button>

              </div>

            )}

          </div>

          {/* ===================================================
              SYSTEM INFORMATION
          =================================================== */}

          <div className="admin-panel">

            <div className="admin-panel-header">

              <div>

                <span className="admin-panel-badge">
                  SYSTEM
                </span>

                <h2>
                  System Information
                </h2>

                <p>
                  Current application configuration.
                </p>

              </div>

            </div>

            <div className="admin-info-list">

              <div className="admin-info-row">
                <span>
                  Application
                </span>

                <strong>
                  ThyroCare AI
                </strong>
              </div>

              <div className="admin-info-row">
                <span>
                  Model
                </span>

                <strong>
                  {modelName}
                </strong>
              </div>

              <div className="admin-info-row">
                <span>
                  Explainability
                </span>

                <strong>
                  SHAP
                </strong>
              </div>

              <div className="admin-info-row">
                <span>
                  Counterfactuals
                </span>

                <strong>
                  {result
                    ? "Available"
                    : "Waiting"}
                </strong>
              </div>

              <div className="admin-info-row">
                <span>
                  Prediction Service
                </span>

                <strong>
                  Configured
                </strong>
              </div>

              <div className="admin-info-row">
                <span>
                  Prediction History
                </span>

                <strong>
                  {historyCount} records
                </strong>
              </div>

            </div>

            <div className="admin-system-status">

              <span className="admin-status-dot" />

              <span>
                ThyroCare AI interface ready
              </span>

            </div>

          </div>

        </section>

        {/* =====================================================
            PERSISTENT RECENT ANALYSIS
        ===================================================== */}

        <section className="admin-panel admin-history-panel">

          <div className="admin-panel-header">

            <div>

              <span className="admin-panel-badge">
                ACTIVITY
              </span>

              <h2>
                Recent Analysis
              </h2>

              <p>
                Prediction activity stored by
                the ThyroCare AI backend.
              </p>

            </div>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                flexWrap: "wrap",
              }}
            >

              <span className="admin-history-count">
                {historyCount}{" "}
                {historyCount === 1
                  ? "analysis"
                  : "analyses"}
              </span>

              <button
                type="button"
                className="admin-secondary-button"
                onClick={loadPredictionHistory}
                disabled={historyLoading}
              >
                {historyLoading
                  ? "Refreshing..."
                  : "↻ Refresh"}
              </button>

            </div>

          </div>

          {historyLoading ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                📋
              </div>

              <h3>
                Loading Prediction History
              </h3>

              <p>
                Fetching stored prediction activity
                from the ThyroCare AI server.
              </p>

            </div>

          ) : historyError ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                ⚠️
              </div>

              <h3>
                History Unavailable
              </h3>

              <p>
                {historyError}
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={loadPredictionHistory}
              >
                Try Again
              </button>

            </div>

          ) : predictionHistory.length === 0 ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                📋
              </div>

              <h3>
                No Analysis History
              </h3>

              <p>
                Completed predictions will
                appear here automatically.
              </p>

            </div>

          ) : (

            <div className="admin-table-wrapper">

              <table className="admin-table">

                <thead>

                  <tr>
                    <th>#</th>
                    <th>Time</th>
                    <th>Prediction</th>
                    <th>Class 0</th>
                    <th>Class 1</th>
                    <th>Model</th>
                  </tr>

                </thead>

                <tbody>

                  {predictionHistory.map(
                    (item, index) => {

                      const itemPrediction =
                        item?.prediction ?? null;

                      const itemClass0 =
                        item?.class_0_probability ??
                        0;

                      const itemClass1 =
                        item?.class_1_probability ??
                        0;

                      const itemModel =
                        item?.model ||
                        "XGBoost";

                      const isPositive =
                        Number(
                          itemPrediction
                        ) === 1;

                      return (
                        <tr
                          key={
                            item?.id ??
                            `${item?.timestamp}-${index}`
                          }
                        >

                          <td>

                            <span className="admin-row-number">
                              {index + 1}
                            </span>

                          </td>

                          <td>

                            <span className="admin-date">
                              {formatDate(
                                item?.timestamp
                              )}
                            </span>

                          </td>

                          <td>

                            <span
                              className={
                                isPositive
                                  ? "admin-badge admin-badge-positive"
                                  : "admin-badge admin-badge-negative"
                              }
                            >
                              {isPositive
                                ? "Class 1"
                                : "Class 0"}
                            </span>

                          </td>

                          <td>
                            {formatPercent(
                              itemClass0
                            )}
                          </td>

                          <td>
                            {formatPercent(
                              itemClass1
                            )}
                          </td>

                          <td>

                            <span className="admin-model-tag">
                              {itemModel}
                            </span>

                          </td>

                        </tr>
                      );
                    }
                  )}

                </tbody>

              </table>

            </div>

          )}

        </section>

        {/* =====================================================
            SHAP
        ===================================================== */}

        <section
          className="admin-panel"
          style={{
            marginTop: "20px",
          }}
        >

          <div className="admin-panel-header">

            <div>

              <span className="admin-panel-badge">
                XAI
              </span>

              <h2>
                Top SHAP Features
              </h2>

              <p>
                Features contributing most to
                the latest model prediction.
              </p>

            </div>

          </div>

          {!result ||
          topShap.length === 0 ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                🧠
              </div>

              <h3>
                SHAP Explanation Waiting
              </h3>

              <p>
                Run an analysis to generate
                model explainability information.
              </p>

            </div>

          ) : (

            <div className="admin-shap-list">

              {topShap.map(
                (item, index) => {

                  const numericValue =
                    Number(item.value) || 0;

                  const magnitude =
                    Math.min(
                      Math.abs(numericValue) * 100,
                      100
                    );

                  return (
                    <div
                      className="admin-shap-item"
                      key={`${item.feature}-${index}`}
                    >

                      <div className="admin-shap-main">

                        <div className="admin-shap-rank">
                          {index + 1}
                        </div>

                        <div className="admin-shap-feature">

                          <span>
                            {item.feature}
                          </span>

                          <div className="admin-shap-bar">

                            <div
                              className="admin-shap-bar-fill"
                              style={{
                                width: `${magnitude}%`,
                              }}
                            />

                          </div>

                        </div>

                        <strong
                          className={
                            numericValue >= 0
                              ? "shap-positive"
                              : "shap-negative"
                          }
                        >
                          {numericValue >= 0
                            ? "+"
                            : ""}
                          {numericValue.toFixed(4)}
                        </strong>

                      </div>

                    </div>
                  );
                }
              )}

            </div>

          )}

        </section>

        {/* =====================================================
            NOTICE
        ===================================================== */}

        <section
          className="admin-panel admin-disclaimer"
          style={{
            marginTop: "20px",
          }}
        >

          <div className="admin-disclaimer-icon">
            ⚕
          </div>

          <div>

            <h2>
              Important Notice
            </h2>

            <p>
              ThyroCare AI provides machine-learning
              predictions and explainability information
              for academic and decision-support purposes.
              The model output is not a medical diagnosis
              and should not replace evaluation by a
              qualified healthcare professional.
            </p>

          </div>

        </section>

      </div>

    </main>
  );
}
