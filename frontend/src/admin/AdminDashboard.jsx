import React, { useMemo } from "react";

/* =========================================================
   THYROCARE AI
   ADMIN DASHBOARD
========================================================= */

export default function AdminDashboard({
  result,
  analysisHistory = [],
  onNewAnalysis,
  onViewResults,
  onLogout,
}) {
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

    /*
     * Backend can return:
     *
     * shap_values: [...]
     *
     * OR
     *
     * shap: {
     *   available: true,
     *   features: [...]
     * }
     */

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
        if (
          typeof item === "number"
        ) {
          return {
            feature:
              `Feature ${index + 1}`,
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
     HISTORY
  ========================================================= */

  const historyCount =
    analysisHistory.length;

  /* =========================================================
     VIEW RESULTS HANDLER
     
     IMPORTANT:
     This explicitly calls the callback supplied
     by App.jsx.
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
     NEW ANALYSIS HANDLER
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
     LOGOUT HANDLER
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
              onClick={
                handleNewAnalysis
              }
            >
              <span>＋</span>
              New Analysis
            </button>

            <button
              type="button"
              className="admin-secondary-button"
              onClick={
                handleViewResults
              }
              disabled={!result}
            >
              View Results
            </button>

            <button
              type="button"
              className="admin-secondary-button admin-danger-button"
              onClick={
                handleLogout
              }
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
                Analyses This Session
              </span>

              <strong className="admin-stat-value">
                {historyCount}
              </strong>

              <span className="admin-stat-subtext">
                Prediction requests
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
                  onClick={
                    handleNewAnalysis
                  }
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
                                ? class0Probability *
                                    100
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
                                ? class1Probability *
                                    100
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

                {/* DIRECT RESULT BUTTON */}

                <button
                  type="button"
                  className="primary-button admin-view-results-button"
                  onClick={
                    handleViewResults
                  }
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
                  Session History
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
            RECENT ANALYSIS
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
                Prediction activity recorded
                during this browser session.
              </p>

            </div>

            <span className="admin-history-count">
              {historyCount}{" "}
              {historyCount === 1
                ? "analysis"
                : "analyses"}
            </span>

          </div>

          {analysisHistory.length === 0 ? (

            <div className="admin-empty">

              <div className="admin-empty-icon">
                📋
              </div>

              <h3>
                No Analysis History
              </h3>

              <p>
                Your completed predictions will
                appear here.
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

                  {analysisHistory.map(
                    (item, index) => {

                      const itemPrediction =
                        item?.prediction ??
                        item?.predicted_class ??
                        item?.class ??
                        null;

                      const itemProbabilities =
                        item?.probabilities ??
                        item?.class_probabilities ??
                        {};

                      const itemClass0 =
                        itemProbabilities?.["0"] ??
                        itemProbabilities?.class_0 ??
                        item?.class0 ??
                        item?.class_0_probability ??
                        0;

                      const itemClass1 =
                        itemProbabilities?.["1"] ??
                        itemProbabilities?.class_1 ??
                        item?.class1 ??
                        item?.class_1_probability ??
                        0;

                      const itemModel =
                        item?.model ??
                        item?.model_name ??
                        item?.algorithm ??
                        "XGBoost";

                      const isPositive =
                        Number(
                          itemPrediction
                        ) === 1;

                      return (
                        <tr
                          key={
                            item?.id ??
                            `${item?.time ?? "analysis"}-${index}`
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
                                item?.time ??
                                  item?.created_at ??
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
                    Number(
                      item.value
                    ) || 0;

                  const magnitude =
                    Math.min(
                      Math.abs(
                        numericValue
                      ) * 100,
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
                          {numericValue.toFixed(
                            4
                          )}
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
