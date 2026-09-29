import React, { useMemo, useState } from "react";
import "./index.css";
import AdminLogin from "./admin/AdminLogin";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

const initialForm = {
  age: 45,
  sex: 1,

  "on thyroxine": 0,
  "query on thyroxine": 0,
  "on antithyroid medication": 0,

  sick: 0,
  pregnant: 0,

  "thyroid surgery": 0,
  "I131 treatment": 0,

  "query hypothyroid": 0,
  "query hyperthyroid": 0,

  lithium: 0,
  goitre: 0,
  tumor: 0,
  hypopituitary: 0,
  psych: 0,

  "TSH measured": 1,
  TSH: 2.5,

  "T3 measured": 1,

  "TT4 measured": 1,
  TT4: 110,

  "T4U measured": 1,
  T4U: 1.0,

  "FTI measured": 1,
  FTI: 110,
};

const binaryFields = [
  ["sex", "Sex"],
  ["on thyroxine", "On thyroxine"],
  ["query on thyroxine", "Query on thyroxine"],
  ["on antithyroid medication", "On antithyroid medication"],
  ["sick", "Sick"],
  ["pregnant", "Pregnant"],
  ["thyroid surgery", "Thyroid surgery"],
  ["I131 treatment", "I131 treatment"],
  ["query hypothyroid", "Query hypothyroid"],
  ["query hyperthyroid", "Query hyperthyroid"],
  ["lithium", "Lithium"],
  ["goitre", "Goitre"],
  ["tumor", "Tumor"],
  ["hypopituitary", "Hypopituitary"],
  ["psych", "Psych"],
  ["TSH measured", "TSH measured"],
  ["T3 measured", "T3 measured"],
  ["TT4 measured", "TT4 measured"],
  ["T4U measured", "T4U measured"],
  ["FTI measured", "FTI measured"],
];

function formatNumber(value, digits = 2) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toFixed(digits);
}

function formatPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return `${(number * 100).toFixed(2)}%`;
}

function getErrorMessage(data, fallback = "Prediction failed.") {
  if (!data) return fallback;

  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item, index) => {
        if (typeof item === "string") {
          return `${index + 1}. ${item}`;
        }

        if (item?.loc && item?.msg) {
          const location = Array.isArray(item.loc)
            ? item.loc.join(" → ")
            : String(item.loc);

          return `${index + 1}. ${location}: ${item.msg}`;
        }

        if (item?.msg) {
          return `${index + 1}. ${item.msg}`;
        }

        return `${index + 1}. ${JSON.stringify(item)}`;
      })
      .join("\n");
  }

  if (typeof data.message === "string") {
    return data.message;
  }

  return fallback;
}

function normalizeShap(shap) {
  if (!shap) return [];

  if (Array.isArray(shap)) {
    return shap
      .map((item) => {
        if (typeof item === "object" && item !== null) {
          return {
            feature:
              item.feature ??
              item.name ??
              item.feature_name ??
              "Feature",

            value: Number(
              item.value ??
                item.shap_value ??
                item.impact ??
                0
            ),
          };
        }

        return null;
      })
      .filter(Boolean)
      .sort(
        (a, b) =>
          Math.abs(b.value) - Math.abs(a.value)
      );
  }

  if (typeof shap === "object") {
    return Object.entries(shap)
      .map(([feature, value]) => ({
        feature,
        value: Number(
          typeof value === "object"
            ? value.value ??
              value.shap_value ??
              value.impact ??
              0
            : value
        ),
      }))
      .filter((item) => Number.isFinite(item.value))
      .sort(
        (a, b) =>
          Math.abs(b.value) - Math.abs(a.value)
      );
  }

  return [];
}

function normalizeCounterfactuals(counterfactuals) {
  if (!counterfactuals) {
    return {
      available: false,
      features: [],
      scenarios: [],
    };
  }

  return {
    available: Boolean(counterfactuals.available),

    features: Array.isArray(
      counterfactuals.features
    )
      ? counterfactuals.features
      : [],

    scenarios: Array.isArray(
      counterfactuals.scenarios
    )
      ? counterfactuals.scenarios
      : [],

    target_prediction:
      counterfactuals.target_prediction,

    target_label:
      counterfactuals.target_label,
  };
}

export default function App() {
  const [page, setPage] = useState("home");

  const [form, setForm] = useState(initialForm);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [menuOpen, setMenuOpen] = useState(false);

  const [adminAuthenticated, setAdminAuthenticated] =
    useState(false);

  const shapValues = useMemo(() => {
    return normalizeShap(
      result?.shap_values ??
        result?.shap ??
        result?.explanation?.shap_values ??
        result?.explanation?.shap
    );
  }, [result]);

  const counterfactuals = useMemo(() => {
    return normalizeCounterfactuals(
      result?.counterfactuals ??
        result?.explanation?.counterfactuals
    );
  }, [result]);

  const prediction = result?.prediction;

  const isDisease =
    prediction === 1 ||
    prediction === "1" ||
    result?.prediction_label ===
      "Thyroid Disease Predicted";

  const predictionLabel = isDisease
    ? "Thyroid Disease Predicted"
    : "Thyroid Disease Not Predicted";

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleBinaryChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: Number(value),
    }));
  };

  const buildPayload = () => {
    return {
      age: Number(form.age),
      sex: Number(form.sex),

      on_thyroxine: Number(
        form["on thyroxine"]
      ),

      query_on_thyroxine: Number(
        form["query on thyroxine"]
      ),

      on_antithyroid_medication: Number(
        form["on antithyroid medication"]
      ),

      sick: Number(form.sick),
      pregnant: Number(form.pregnant),

      thyroid_surgery: Number(
        form["thyroid surgery"]
      ),

      I131_treatment: Number(
        form["I131 treatment"]
      ),

      query_hypothyroid: Number(
        form["query hypothyroid"]
      ),

      query_hyperthyroid: Number(
        form["query hyperthyroid"]
      ),

      lithium: Number(form.lithium),
      goitre: Number(form.goitre),
      tumor: Number(form.tumor),
      hypopituitary: Number(
        form.hypopituitary
      ),
      psych: Number(form.psych),

      TSH_measured: Number(
        form["TSH measured"]
      ),

      TSH: Number(form.TSH),

      T3_measured: Number(
        form["T3 measured"]
      ),

      TT4_measured: Number(
        form["TT4 measured"]
      ),

      TT4: Number(form.TT4),

      T4U_measured: Number(
        form["T4U measured"]
      ),

      T4U: Number(form.T4U),

      FTI_measured: Number(
        form["FTI measured"]
      ),

      FTI: Number(form.FTI),
    };
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = buildPayload();

      const controller =
        new AbortController();

      const timeout = setTimeout(() => {
        controller.abort();
      }, 180000);

      let response;

      try {
        response = await fetch(
          `${API_URL}/predict`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",

              Accept:
                "application/json",
            },

            body: JSON.stringify(payload),

            signal: controller.signal,
          }
        );
      } finally {
        clearTimeout(timeout);
      }

      let data = null;

      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok) {
        throw new Error(
          getErrorMessage(
            data,
            `Server returned HTTP ${response.status}.`
          )
        );
      }

      setResult(data);

      setPage("results");

    } catch (err) {
      console.error(
        "Prediction error:",
        err
      );

      if (
        err?.name ===
        "AbortError"
      ) {
        setError(
          "The prediction server took too long to respond. Please try again."
        );
      } else if (
        err?.message?.includes(
          "Failed to fetch"
        )
      ) {
        setError(
          "Unable to connect to the prediction server."
        );
      } else {
        setError(
          err?.message ||
            "An unexpected prediction error occurred."
        );
      }

    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
    setPage("prediction");
  };

  const goTo = (target) => {
    setPage(target);
    setMenuOpen(false);

    if (target !== "results") {
      setError("");
    }
  };

  const downloadReport = () => {
    if (!result) return;

    const report = `
THYROCARE — AI THYROID ANALYSIS
================================

MODEL
-----
XGBoost

PREDICTION
----------
${predictionLabel}

Class 0 Probability:
${formatPercent(
  result.class_0_probability
)}

Class 1 Probability:
${formatPercent(
  result.class_1_probability
)}

SHAP EXPLANATION
----------------
${
  shapValues.length
    ? shapValues
        .slice(0, 15)
        .map(
          (item, index) =>
            `${index + 1}. ${
              item.feature
            }: ${formatNumber(
              item.value,
              4
            )}`
        )
        .join("\n")
    : "SHAP values unavailable."
}

COUNTERFACTUAL ANALYSIS
-----------------------
${
  counterfactuals.available
    ? counterfactuals.scenarios
        .map((scenario, index) => {
          const changes =
            Array.isArray(
              scenario.changes
            )
              ? scenario.changes
                  .map(
                    (change) =>
                      `${change.feature}: ${change.original_value} -> ${change.counterfactual_value}`
                  )
                  .join(", ")
              : "No changes listed";

          return `
Scenario ${index + 1}
${changes}

Prediction:
${
  scenario.prediction_label ??
  scenario.prediction
}

Class 0:
${formatPercent(
  scenario.class_0_probability
)}

Class 1:
${formatPercent(
  scenario.class_1_probability
)}
`;
        })
        .join("\n")
    : "Counterfactual analysis unavailable."
}

DISCLAIMER
----------
This system is an academic machine-learning project.
It is not a medical diagnosis and should not replace
professional medical advice.

Generated by ThyroCare AI.
`;

    const blob = new Blob(
      [report],
      {
        type:
          "text/plain;charset=utf-8",
      }
    );

    const url =
      URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;

    link.download =
      "thyrocare-analysis-report.txt";

    document.body.appendChild(link);

    link.click();

    link.remove();

    URL.revokeObjectURL(url);
  };

  const renderBinaryField = (
    name,
    label
  ) => {
    return (
      <div
        className="field"
        key={name}
      >
        <label>{label}</label>

        <select
          name={name}
          value={form[name]}
          onChange={
            handleBinaryChange
          }
        >
          <option value={0}>
            No / 0
          </option>

          <option value={1}>
            Yes / 1
          </option>
        </select>
      </div>
    );
  };

  // ======================================================
  // HOME
  // ======================================================

  const renderHomePage = () => {
    return (
      <main className="home-page">

        <section className="hero">

          <div className="hero-content">

            <span className="eyebrow">
              EXPLAINABLE AI • THYROID ANALYSIS
            </span>

            <h1>
              Smarter
              <br />
              <span>
                Thyroid Analysis.
              </span>
            </h1>

            <p>
              An AI-powered thyroid disease
              prediction system using
              XGBoost, SHAP explanations,
              and counterfactual analysis.
            </p>

            <div className="hero-actions">

              <button
                className="primary-button"
                onClick={() =>
                  goTo("prediction")
                }
              >
                Start Analysis
                <span>→</span>
              </button>

              <button
                className="secondary-button"
                onClick={() =>
                  goTo("about")
                }
              >
                Learn More
              </button>

            </div>

          </div>

          <div className="hero-visual">

            <div className="orb">

              <div className="orb-inner">

                <span>AI</span>

                <small>
                  XGBoost
                </small>

              </div>

            </div>

            <div className="floating-card card-one">
              <span>
                Accuracy
              </span>

              <strong>
                99.87%
              </strong>
            </div>

            <div className="floating-card card-two">
              <span>
                Explainable
              </span>

              <strong>
                SHAP + CF
              </strong>
            </div>

          </div>

        </section>

        <section className="feature-section">

          <div className="section-intro">

            <span className="eyebrow">
              HOW IT WORKS
            </span>

            <h2>
              From patient data
              <br />
              to explainable prediction.
            </h2>

          </div>

          <div className="feature-grid">

            <div className="feature-card">
              <span>01</span>

              <h3>
                Input
              </h3>

              <p>
                Enter patient and
                thyroid measurement
                values.
              </p>
            </div>

            <div className="feature-card">
              <span>02</span>

              <h3>
                Prediction
              </h3>

              <p>
                XGBoost analyzes
                the 25 clinical
                features.
              </p>
            </div>

            <div className="feature-card">
              <span>03</span>

              <h3>
                Explanation
              </h3>

              <p>
                SHAP and
                counterfactual
                analysis explain
                the prediction.
              </p>
            </div>

          </div>

        </section>

      </main>
    );
  };

  // ======================================================
  // PREDICTION
  // ======================================================

  const renderPredictionPage = () => {
    return (
      <main className="page prediction-page">

        <div className="page-header">

          <div>

            <span className="eyebrow">
              AI PREDICTION
            </span>

            <h1>
              Analyze Thyroid
            </h1>

            <p>
              Enter the patient measurements
              and clinical indicators below.
            </p>

          </div>

          <div className="model-badge">

            <span className="status-dot"></span>

            XGBoost Model

          </div>

        </div>

        {error && (
          <div className="error-box">

            <div className="error-title">
              Prediction Error
            </div>

            <div className="error-message">
              {error}
            </div>

          </div>
        )}

        <section className="form-card">

          <div className="section-heading">

            <div>

              <span className="section-number">
                01
              </span>

              <h2>
                Patient Information
              </h2>

            </div>

          </div>

          <div className="form-grid">

            <div className="field">

              <label>
                Age
              </label>

              <input
                type="number"
                name="age"
                value={form.age}
                min="1"
                max="120"
                onChange={
                  handleChange
                }
              />

            </div>

            {renderBinaryField(
              "sex",
              "Sex"
            )}

          </div>

        </section>

        <section className="form-card">

          <div className="section-heading">

            <div>

              <span className="section-number">
                02
              </span>

              <h2>
                Clinical Indicators
              </h2>

            </div>

          </div>

          <div className="form-grid">

            {binaryFields
              .slice(1, 15)
              .map(
                ([
                  name,
                  label,
                ]) =>
                  renderBinaryField(
                    name,
                    label
                  )
              )}

          </div>

        </section>

        <section className="form-card">

          <div className="section-heading">

            <div>

              <span className="section-number">
                03
              </span>

              <h2>
                Thyroid Measurements
              </h2>

            </div>

          </div>

          <div className="form-grid">

            <div className="field">

              <label>
                TSH
              </label>

              <input
                type="number"
                name="TSH"
                value={form.TSH}
                step="0.01"
                onChange={
                  handleChange
                }
              />

            </div>

            <div className="field">

              <label>
                TT4
              </label>

              <input
                type="number"
                name="TT4"
                value={form.TT4}
                step="0.01"
                onChange={
                  handleChange
                }
              />

            </div>

            <div className="field">

              <label>
                T4U
              </label>

              <input
                type="number"
                name="T4U"
                value={form.T4U}
                step="0.01"
                onChange={
                  handleChange
                }
              />

            </div>

            <div className="field">

              <label>
                FTI
              </label>

              <input
                type="number"
                name="FTI"
                value={form.FTI}
                step="0.01"
                onChange={
                  handleChange
                }
              />

            </div>

          </div>

        </section>

        <section className="form-card">

          <div className="section-heading">

            <div>

              <span className="section-number">
                04
              </span>

              <h2>
                Measurement Availability
              </h2>

            </div>

          </div>

          <div className="form-grid">

            {[
              [
                "TSH measured",
                "TSH measured",
              ],
              [
                "T3 measured",
                "T3 measured",
              ],
              [
                "TT4 measured",
                "TT4 measured",
              ],
              [
                "T4U measured",
                "T4U measured",
              ],
              [
                "FTI measured",
                "FTI measured",
              ],
            ].map(
              ([
                name,
                label,
              ]) =>
                renderBinaryField(
                  name,
                  label
                )
            )}

          </div>

        </section>

        <div className="action-area">

          <button
            className="primary-button"
            onClick={
              handleAnalyze
            }
            disabled={loading}
          >

            {loading ? (
              <>
                <span className="spinner"></span>

                Analyzing...
              </>
            ) : (
              <>
                Analyze Thyroid

                <span>
                  →
                </span>
              </>
            )}

          </button>

          <button
            className="secondary-button"
            onClick={
              resetForm
            }
            disabled={loading}
          >
            Reset
          </button>

        </div>

        <div className="process-strip">

          <div>
            <strong>
              01
            </strong>

            <span>
              Input
            </span>
          </div>

          <div className="process-line"></div>

          <div>
            <strong>
              02
            </strong>

            <span>
              XGBoost Prediction
            </span>
          </div>

          <div className="process-line"></div>

          <div>
            <strong>
              03
            </strong>

            <span>
              SHAP + Counterfactuals
            </span>
          </div>

        </div>

      </main>
    );
  };

  // ======================================================
  // RESULTS
  // ======================================================

  const renderResultsPage = () => {

    if (!result) {
      return (
        <main className="page empty-page">

          <div className="empty-card">

            <div className="empty-icon">
              ⌁
            </div>

            <h1>
              No Analysis Yet
            </h1>

            <p>
              Enter patient information
              to generate an AI-powered
              thyroid analysis.
            </p>

            <button
              className="primary-button"
              onClick={() =>
                goTo("prediction")
              }
            >
              Start Analysis →
            </button>

          </div>

        </main>
      );
    }

    return (
      <main className="page results-page">

        <div className="page-header">

          <div>

            <span className="eyebrow">
              ANALYSIS COMPLETE
            </span>

            <h1>
              Prediction Results
            </h1>

            <p>
              XGBoost prediction with
              explainable AI analysis.
            </p>

          </div>

          <div className="result-actions">

            <button
              className="secondary-button"
              onClick={
                downloadReport
              }
            >
              Download Report
            </button>

            <button
              className="secondary-button"
              onClick={
                resetForm
              }
            >
              New Analysis
            </button>

          </div>

        </div>

        <section
          className={`prediction-result ${
            isDisease
              ? "positive"
              : "negative"
          }`}
        >

          <div className="result-icon">
            {isDisease
              ? "!"
              : "✓"}
          </div>

          <div className="result-main">

            <span>
              MODEL PREDICTION
            </span>

            <h2>
              {predictionLabel}
            </h2>

            <p>
              XGBoost classification
              result based on the
              entered clinical features.
            </p>

          </div>

          <div className="result-class">

            <small>
              CLASS
            </small>

            <strong>
              {isDisease
                ? "1"
                : "0"}
            </strong>

          </div>

        </section>

        <section className="probability-grid">

          <div className="metric-card">

            <span>
              Class 0
            </span>

            <strong>
              {formatPercent(
                result.class_0_probability
              )}
            </strong>

            <small>
              Not Predicted
            </small>

            <div className="progress">

              <div
                style={{
                  width: `${Math.min(
                    Number(
                      result.class_0_probability
                    ) * 100,
                    100
                  )}%`,
                }}
              />

            </div>

          </div>

          <div className="metric-card">

            <span>
              Class 1
            </span>

            <strong>
              {formatPercent(
                result.class_1_probability
              )}
            </strong>

            <small>
              Thyroid Disease Predicted
            </small>

            <div className="progress">

              <div
                style={{
                  width: `${Math.min(
                    Number(
                      result.class_1_probability
                    ) * 100,
                    100
                  )}%`,
                }}
              />

            </div>

          </div>

        </section>

        <section className="analysis-card">

          <div className="card-title">

            <div>

              <span className="eyebrow">
                EXPLAINABLE AI
              </span>

              <h2>
                SHAP Feature Impact
              </h2>

            </div>

            <span className="info-pill">
              SHAP
            </span>

          </div>

          {shapValues.length ===
          0 ? (
            <div className="no-data">
              SHAP explanation is not
              available for this prediction.
            </div>
          ) : (
            <div className="shap-list">

              {shapValues
                .slice(0, 12)
                .map(
                  (
                    item,
                    index
                  ) => {

                    const magnitude =
                      Math.min(
                        Math.abs(
                          item.value
                        ) * 20,
                        100
                      );

                    const positive =
                      item.value >= 0;

                    return (
                      <div
                        className="shap-row"
                        key={`${item.feature}-${index}`}
                      >

                        <div className="shap-name">

                          <span>
                            {index + 1}
                          </span>

                          <strong>
                            {
                              item.feature
                            }
                          </strong>

                        </div>

                        <div className="shap-bar">

                          <div
                            className={
                              positive
                                ? "shap-positive"
                                : "shap-negative"
                            }
                            style={{
                              width: `${Math.max(
                                magnitude,
                                2
                              )}%`,
                            }}
                          />

                        </div>

                        <div
                          className={`shap-value ${
                            positive
                              ? "value-positive"
                              : "value-negative"
                          }`}
                        >
                          {item.value >
                          0
                            ? "+"
                            : ""}

                          {formatNumber(
                            item.value,
                            4
                          )}
                        </div>

                      </div>
                    );
                  }
                )}

            </div>
          )}

          <div className="shap-legend">

            <span>
              <i className="legend-positive"></i>
              Positive contribution
            </span>

            <span>
              <i className="legend-negative"></i>
              Negative contribution
            </span>

          </div>

        </section>

        <section className="analysis-card">

          <div className="card-title">

            <div>

              <span className="eyebrow">
                WHAT-IF ANALYSIS
              </span>

              <h2>
                Counterfactual Explanations
              </h2>

            </div>

            <span className="info-pill">
              DiCE / CF
            </span>

          </div>

          <p className="card-description">
            Counterfactual analysis shows
            how changing selected input
            values can alter the model's
            prediction.
          </p>

          {!counterfactuals.available ? (
            <div className="no-data">
              Counterfactual analysis is
              currently unavailable.
            </div>
          ) : (
            <>
              {counterfactuals
                .features.length >
                0 && (
                <div className="cf-summary">

                  <h3>
                    Suggested Changes
                  </h3>

                  {counterfactuals.features.map(
                    (
                      feature,
                      index
                    ) => (
                      <div
                        className="cf-change"
                        key={`${feature.feature}-${index}`}
                      >

                        <div>

                          <strong>
                            {
                              feature.feature
                            }
                          </strong>

                          <span>
                            Original:{" "}
                            {
                              feature.original_value
                            }
                          </span>

                        </div>

                        <div className="cf-arrow">
                          →
                        </div>

                        <div>

                          <strong>
                            {
                              feature.counterfactual_value
                            }
                          </strong>

                          <span>
                            Suggested value
                          </span>

                        </div>

                      </div>
                    )
                  )}

                </div>
              )}

              <div className="scenario-grid">

                {counterfactuals
                  .scenarios
                  .map(
                    (
                      scenario,
                      index
                    ) => (

                      <div
                        className="scenario-card"
                        key={
                          scenario.scenario ??
                          index
                        }
                      >

                        <div className="scenario-top">

                          <span>
                            Scenario{" "}
                            {
                              scenario.scenario ??
                              index + 1
                            }
                          </span>

                          <strong>
                            Class{" "}
                            {
                              scenario.prediction
                            }
                          </strong>

                        </div>

                        {Array.isArray(
                          scenario.changes
                        ) &&
                          scenario
                            .changes
                            .length >
                            0 && (
                            <div className="scenario-changes">

                              {scenario.changes.map(
                                (
                                  change,
                                  changeIndex
                                ) => (

                                  <div
                                    key={
                                      changeIndex
                                    }
                                  >

                                    <span>
                                      {
                                        change.feature
                                      }
                                    </span>

                                    <strong>
                                      {
                                        change.original_value
                                      }{" "}
                                      →{" "}
                                      {
                                        change.counterfactual_value
                                      }
                                    </strong>

                                  </div>

                                )
                              )}

                            </div>
                          )}

                        <div className="scenario-probabilities">

                          <div>

                            <span>
                              Class 0
                            </span>

                            <strong>
                              {formatPercent(
                                scenario.class_0_probability
                              )}
                            </strong>

                          </div>

                          <div>

                            <span>
                              Class 1
                            </span>

                            <strong>
                              {formatPercent(
                                scenario.class_1_probability
                              )}
                            </strong>

                          </div>

                        </div>

                      </div>

                    )
                  )}

              </div>

            </>
          )}

        </section>

        <section className="disclaimer">

          <strong>
            Important
          </strong>

          <p>
            ThyroCare is an academic
            machine-learning project.
            This prediction is not a
            medical diagnosis and should
            not replace evaluation by a
            qualified healthcare
            professional.
          </p>

        </section>

      </main>
    );
  };

  // ======================================================
  // ABOUT
  // ======================================================

  const renderAboutPage = () => {
    return (
      <main className="page about-page">

        <div className="page-header">

          <div>

            <span className="eyebrow">
              ABOUT THE PROJECT
            </span>

            <h1>
              ThyroCare
            </h1>

            <p>
              Enhancing Thyroid Disease
              Diagnosis With Machine Learning
              and Counterfactual Explainable AI.
            </p>

          </div>

        </div>

        <section className="about-grid">

          <div className="about-card large">

            <span className="eyebrow">
              PROJECT OVERVIEW
            </span>

            <h2>
              Machine learning with
              transparent explanations.
            </h2>

            <p>
              ThyroCare is a final-year
              B.Tech project designed to
              demonstrate how machine
              learning can be combined
              with Explainable AI techniques
              for thyroid disease prediction.
            </p>

            <p>
              The system uses an XGBoost
              classification model and
              provides SHAP-based feature
              explanations together with
              counterfactual scenarios.
            </p>

          </div>

          <div className="about-card">

            <span className="eyebrow">
              MODEL
            </span>

            <h2>
              XGBoost
            </h2>

            <p>
              Gradient-boosted decision
              trees trained on thyroid
              clinical features.
            </p>

          </div>

          <div className="about-card">

            <span className="eyebrow">
              EXPLANATION
            </span>

            <h2>
              SHAP
            </h2>

            <p>
              Shows how individual
              features contribute to
              the prediction.
            </p>

          </div>

          <div className="about-card">

            <span className="eyebrow">
              WHAT-IF ANALYSIS
            </span>

            <h2>
              Counterfactuals
            </h2>

            <p>
              Demonstrates how changing
              input features can affect
              model predictions.
            </p>

          </div>

        </section>

        <section className="disclaimer">

          <strong>
            Academic Project
          </strong>

          <p>
            This application is intended
            for educational and research
            demonstration purposes.
            It is not a medical diagnostic
            device.
          </p>

        </section>

      </main>
    );
  };

  // ======================================================
  // ADMIN DASHBOARD
  // ======================================================

  const renderAdminDashboard = () => {
    return (
      <main className="page admin-dashboard">

        <div className="page-header">

          <div>

            <span className="eyebrow">
              ADMINISTRATION
            </span>

            <h1>
              Admin Dashboard
            </h1>

            <p>
              Manage datasets, machine-learning
              models, evaluation and prediction
              history.
            </p>

          </div>

          <button
            className="secondary-button"
            onClick={() => {
              setAdminAuthenticated(
                false
              );

              setPage("home");
            }}
          >
            Logout
          </button>

        </div>

        <section className="admin-welcome">

          <div className="admin-status">

            <span className="status-dot"></span>

            Administrator authenticated

          </div>

          <h2>
            ThyroCare AI Control Center
          </h2>

          <p>
            The administration modules will
            be connected to the FastAPI backend
            in the next stages.
          </p>

        </section>

        <section className="feature-grid">

          <div className="feature-card">

            <span>
              01
            </span>

            <h3>
              📁 Dataset Management
            </h3>

            <p>
              Upload CSV datasets and inspect
              dataset information.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

          <div className="feature-card">

            <span>
              02
            </span>

            <h3>
              ⚙️ Dataset Preprocessing
            </h3>

            <p>
              Clean, preprocess and prepare
              datasets for machine learning.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

          <div className="feature-card">

            <span>
              03
            </span>

            <h3>
              🤖 Algorithm Training
            </h3>

            <p>
              Train Logistic Regression,
              Decision Tree, Random Forest,
              SVM and XGBoost.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

          <div className="feature-card">

            <span>
              04
            </span>

            <h3>
              🎯 Test Accuracy
            </h3>

            <p>
              View accuracy, precision,
              recall, F1-score and ROC-AUC.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

          <div className="feature-card">

            <span>
              05
            </span>

            <h3>
              📊 Algorithm Comparison
            </h3>

            <p>
              Compare trained algorithms
              using interactive charts.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

          <div className="feature-card">

            <span>
              06
            </span>

            <h3>
              👥 Prediction History
            </h3>

            <p>
              View previous prediction
              records and analysis results.
            </p>

            <button
              className="secondary-button"
              disabled
            >
              Coming Next
            </button>

          </div>

        </section>

      </main>
    );
  };

  // ======================================================
  // MAIN NAVIGATION
  // ======================================================

  return (
    <div className="app">

      <header className="navbar">

        <button
          className="brand"
          onClick={() =>
            goTo("home")
          }
        >

          <span className="brand-mark">
            T
          </span>

          <span>

            <strong>
              ThyroCare
            </strong>

            <small>
              AI THYROID ANALYSIS
            </small>

          </span>

        </button>

        <nav
          className={`nav-links ${
            menuOpen
              ? "open"
              : ""
          }`}
        >

          <button
            className={
              page === "home"
                ? "active"
                : ""
            }
            onClick={() =>
              goTo("home")
            }
          >
            Home
          </button>

          <button
            className={
              page === "prediction"
                ? "active"
                : ""
            }
            onClick={() =>
              goTo("prediction")
            }
          >
            Analyze
          </button>

          <button
            className={
              page === "results"
                ? "active"
                : ""
            }
            onClick={() =>
              goTo("results")
            }
          >
            Results
          </button>

          <button
            className={
              page === "about"
                ? "active"
                : ""
            }
            onClick={() =>
              goTo("about")
            }
          >
            About
          </button>

          <button
            className={
              page === "admin" ||
              page ===
                "admin-dashboard"
                ? "active"
                : ""
            }
            onClick={() =>
              goTo("admin")
            }
          >
            🔐 Admin
          </button>

        </nav>

        <button
          className="menu-button"
          onClick={() =>
            setMenuOpen(
              (previous) =>
                !previous
            )
          }
        >
          ☰
        </button>

      </header>

      {page === "home" &&
        renderHomePage()}

      {page === "prediction" &&
        renderPredictionPage()}

      {page === "results" &&
        renderResultsPage()}

      {page === "about" &&
        renderAboutPage()}

      {page === "admin" && (
        <AdminLogin
          onLogin={() => {
            setAdminAuthenticated(
              true
            );

            setPage(
              "admin-dashboard"
            );
          }}
          onBack={() =>
            goTo("home")
          }
        />
      )}

      {page ===
        "admin-dashboard" &&
        adminAuthenticated &&
        renderAdminDashboard()}

      <footer className="footer">

        <div>

          <strong>
            ThyroCare
          </strong>

          <span>
            AI Thyroid Analysis
          </span>

        </div>

        <p>
          Final Year B.Tech Project •
          Machine Learning &
          Explainable AI
        </p>

      </footer>

    </div>
  );
      }
