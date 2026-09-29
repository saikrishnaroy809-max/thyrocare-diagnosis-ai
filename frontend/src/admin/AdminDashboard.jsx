import React from "react";

export default function AdminDashboard({
  result,
  analysisHistory = [],
  onNewAnalysis,
  onViewResults,
  onLogout,
}) {
  /* =========================================================
     HELPERS
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

  const formatPercent = (value) => {
    const number = Number(value);

    if (Number.isNaN(number)) {
      return "0.00%";
    }

    const percentage =
      number <= 1 ? number * 100 : number;

    return `${percentage.toFixed(2)}%`;
  };

  const formatDate = (value) => {
    if (!value) return "—";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
      return value;
    }

    return date.toLocaleString();
  };

  const predictionLabel =
    Number(prediction) === 1
      ? "Thyroid Disease Predicted"
      : Number(prediction) === 0
      ? "Thyroid Disease Not Predicted"
      : "No prediction yet";

  /* =========================================================
     SHAP DATA
     ========================================================= */

  const shapValues =
    result?.shap_values ??
    result?.shap ??
    result?.feature_importance ??
    [];

  const normalizeShap = () => {
    if (!Array.isArray(shapValues)) {
      return [];
    }

    return shapValues
      .map((item, index) => {
        if (typeof item === "number") {
          return {
            feature: `Feature ${index + 1}`,
            value: item,
          };
        }

        return {
          feature:
            item?.feature ??
            item?.name ??
            item?.feature_name ??
            `Feature ${index + 1}`,
          value:
            Number(
              item?.shap_value ??
                item?.value ??
                item?.impact ??
                0
            ),
        };
      })
      .sort(
        (a, b) =>
          Math.abs(b.value) - Math.abs(a.value)
      )
      .slice(0, 5);
  };

  const topShap = normalizeShap();

  /* =========================================================
     DASHBOARD
     ========================================================= */

  return (
    <main className="admin-dashboard">
      <div className="admin-dashboard-container">

        {/* =====================================================
            TOP BAR
            ===================================================== */}

        <section className="admin-topbar">

          <div>
            <h1>Admin Dashboard</h1>

            <p>
              Monitor ThyroCare AI prediction activity
              and model responses.
            </p>
          </div>

          <div className="admin-topbar-actions">

            <button
              type="button"
              className="admin-secondary-button"
              onClick={onNewAnalysis}
            >
              + New Analysis
            </button>

            <button
              type="button"
              className="admin-secondary-button"
              onClick={onViewResults}
              disabled={!result}
            >
              View Results
            </button>

            <button
              type="button"
              className="admin-secondary-button admin-danger-button"
              onClick={onLogout}
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

            <span className="admin-stat-label">
              Analyses This Session
            </span>

            <strong className="admin-stat-value">
              {analysisHistory.length}
            </strong>

            <span className="admin-stat-subtext">
              Prediction requests
            </span>

          </div>

          <div className="admin-stat-card">

            <span className="admin-stat-label">
              Current Model
            </span>

            <strong className="admin-stat-value">
              {modelName}
            </strong>

            <span className="admin-stat-subtext">
              Machine learning model
            </span>

          </div>

          <div className="admin-stat-card">

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

          <div className="admin-stat-card">

            <span className="admin-stat-label">
              API Status
            </span>

            <strong className="admin-stat-value">
              Online
            </strong>

            <span className="admin-stat-subtext">
              Prediction service configured
            </span>

          </div>

        </section>

        {/* =====================================================
            MAIN CONTENT
            ===================================================== */}

        <section className="admin-content-grid">

          {/* ===================================================
              LATEST PREDICTION
              =================================================== */}

          <div className="admin-panel">

            <div className="admin-panel-header">

              <div>
                <h2>Latest Analysis</h2>

                <p>
                  Most recent model prediction
                </p>
              </div>

              <span className="admin-status">
                <span className="admin-status-dot"></span>
                System Ready
              </span>

            </div>

            {!result ? (
              <div className="admin-empty">

                <p>
                  No prediction has been performed yet.
                </p>

                <br />

                <button
                  type="button"
                  className="primary-button"
                  onClick={onNewAnalysis}
                >
                  Start Analysis
                </button>

              </div>
            ) : (
              <div className="admin-prediction-box">

                <span className="prediction-label">
                  Prediction
                </span>

                <h3>
                  {predictionLabel}
                </h3>

                <div className="admin-probabilities">

                  <div className="admin-probability">

                    <span>
                      Class 0
                    </span>

                    <strong>
                      {formatPercent(
                        class0Probability
                      )}
                    </strong>

                  </div>

                  <div className="admin-probability">

                    <span>
                      Class 1
                    </span>

                    <strong>
                      {formatPercent(
                        class1Probability
                      )}
                    </strong>

                  </div>

                </div>

              </div>
            )}

          </div>

          {/* ===================================================
              SYSTEM INFORMATION
              =================================================== */}

          <div className="admin-panel">

            <div className="admin-panel-header">

              <div>
                <h2>System Information</h2>

                <p>
                  Current application status
                </p>
              </div>

            </div>

            <div className="admin-info-list">

              <div className="admin-info-row">
                <span>Application</span>
                <span>ThyroCare AI</span>
              </div>

              <div className="admin-info-row">
                <span>Model</span>
                <span>{modelName}</span>
              </div>

              <div className="admin-info-row">
                <span>Explainability</span>
                <span>SHAP</span>
              </div>

              <div className="admin-info-row">
                <span>Counterfactuals</span>
                <span>
                  {result
                    ? "Available"
                    : "Waiting"}
                </span>
              </div>

              <div className="admin-info-row">
                <span>Prediction API</span>
                <span>Connected</span>
              </div>

            </div>

          </div>

        </section>

        {/* =====================================================
            RECENT ANALYSIS
            ===================================================== */}

        <section className="admin-panel">

          <div className="admin-panel-header">

            <div>
              <h2>Recent Analysis</h2>

              <p>
                Prediction activity during this session
              </p>
            </div>

          </div>

          {analysisHistory.length === 0 ? (
            <div className="admin-empty">
              No analysis history available yet.
            </div>
          ) : (
            <div className="admin-table-wrapper">

              <table className="admin-table">

                <thead>
                  <tr>
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

                      return (
                        <tr key={item?.id ?? index}>

                          <td>
                            {formatDate(
                              item?.time ??
                                item?.created_at ??
                                item?.timestamp
                            )}
                          </td>

                          <td>

                            <span
                              className={
                                Number(
                                  itemPrediction
                                ) === 1
                                  ? "admin-badge admin-badge-positive"
                                  : "admin-badge admin-badge-negative"
                              }
                            >
                              {Number(
                                itemPrediction
                              ) === 1
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
                            {item?.model ??
                              item?.model_name ??
                              "XGBoost"}
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
            SHAP FEATURES
            ===================================================== */}

        <section
          className="admin-panel"
          style={{ marginTop: "20px" }}
        >

          <div className="admin-panel-header">

            <div>
              <h2>
                Top SHAP Features
              </h2>

              <p>
                Features contributing most to
                the latest model prediction
              </p>
            </div>

          </div>

          {!result || topShap.length === 0 ? (
            <div className="admin-empty">
              SHAP explanation will appear after
              running an analysis.
            </div>
          ) : (
            <div className="admin-shap-list">

              {topShap.map((item, index) => (

                <div
                  className="admin-shap-item"
                  key={`${item.feature}-${index}`}
                >

                  <span>
                    {item.feature}
                  </span>

                  <strong>
                    {Number(item.value) >= 0
                      ? "+"
                      : ""}
                    {Number(item.value).toFixed(4)}
                  </strong>

                </div>

              ))}

            </div>
          )}

        </section>

        {/* =====================================================
            MEDICAL DISCLAIMER
            ===================================================== */}

        <section
          className="admin-panel"
          style={{ marginTop: "20px" }}
        >

          <div className="admin-panel-header">
            <div>
              <h2>
                Important Notice
              </h2>
            </div>
          </div>

          <p
            style={{
              color: "#94a3b8",
              lineHeight: "1.8",
              fontSize: "0.85rem",
            }}
          >
            ThyroCare AI provides machine-learning
            predictions for academic and decision-support
            purposes. The prediction is not a medical
            diagnosis and should not replace evaluation
            by a qualified healthcare professional.
          </p>

        </section>

      </div>
    </main>
  );
        }
