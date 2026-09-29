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
  T4U: 1,
  "FTI measured": 1,
  FTI: 110,
};

const binaryFields = [
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
  "T3 measured",
  "TT4 measured",
  "T4U measured",
  "FTI measured",
];

const numericFields = [
  "age",
  "TSH",
  "TT4",
  "T4U",
  "FTI",
];

function formatNumber(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "0";
  }

  return n.toFixed(2);
}

function formatPercent(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "0.00%";
  }

  return `${(n * 100).toFixed(2)}%`;
}

function getErrorMessage(error) {
  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error?.message) {
    return error.message;
  }

  return "Unable to connect to prediction server.";
}

function normalizeShap(shap) {
  if (!shap) return [];

  if (Array.isArray(shap)) {
    return shap.map((item) => {
      if (typeof item === "object") {
        return {
          feature:
            item.feature ||
            item.name ||
            item.column ||
            "Feature",
          value:
            Number(
              item.value ??
              item.shap_value ??
              item.impact ??
              0
            ),
        };
      }

      return {
        feature: "Feature",
        value: Number(item) || 0,
      };
    });
  }

  if (typeof shap === "object") {
    return Object.entries(shap).map(([feature, value]) => ({
      feature,
      value: Number(value) || 0,
    }));
  }

  return [];
}

function normalizeCounterfactuals(counterfactuals) {
  if (!Array.isArray(counterfactuals)) {
    return [];
  }

  return counterfactuals.map((item) => ({
    feature:
      item.feature ||
      item.changed_feature ||
      "Feature",
    from:
      item.from ??
      item.original ??
      item.old_value ??
      "-",
    to:
      item.to ??
      item.new_value ??
      item.changed_to ??
      "-",
    prediction:
      item.prediction ??
      item.class ??
      item.target ??
      "-",
    probability:
      item.probability ??
      item.class_probability ??
      item.prob ??
      null,
  }));
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

  const goTo = (nextPage) => {
    setPage(nextPage);
    setMenuOpen(false);
    setError("");
  };

  const updateField = (field, value) => {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const buildPayload = () => {
    return {
      age: Number(form.age),
      sex: Number(form.sex),

      on_thyroxine: Number(form["on thyroxine"]),
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
      hypopituitary: Number(form.hypopituitary),
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

    try {
      const controller = new AbortController();

      const timeout = setTimeout(() => {
        controller.abort();
      }, 180000);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(buildPayload()),
          signal: controller.signal,
        }
      );

      clearTimeout(timeout);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
          "Prediction request failed."
        );
      }

      setResult(data);
      setPage("results");
    } catch (err) {
      if (err.name === "AbortError") {
        setError(
          "Prediction request timed out. Please try again."
        );
      } else {
        setError(getErrorMessage(err));
      }
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  const shapData = useMemo(() => {
    return normalizeShap(
      result?.shap_values ||
      result?.shap ||
      result?.explanation
    );
  }, [result]);

  const counterfactualData = useMemo(() => {
    return normalizeCounterfactuals(
      result?.counterfactuals ||
      result?.counterfactual_explanations ||
      []
    );
  }, [result]);

  const predictionClass =
    result?.prediction ??
    result?.predicted_class ??
    result?.class ??
    null;

  const probabilities =
    result?.probabilities ||
    result?.class_probabilities ||
    {};

  const class0 =
    probabilities?.["0"] ??
    probabilities?.class_0 ??
    probabilities?.not_disease ??
    result?.class_0_probability ??
    0;

  const class1 =
    probabilities?.["1"] ??
    probabilities?.class_1 ??
    probabilities?.disease ??
    result?.class_1_probability ??
    0;

  const predictionLabel =
    predictionClass === 1
      ? "Thyroid Disease Predicted"
      : predictionClass === 0
      ? "Thyroid Disease Not Predicted"
      : "Prediction Result";

  const renderNavbar = () => {
    return (
      <nav className="navbar">
        <div
          className="brand"
          onClick={() => goTo("home")}
        >
          <div className="brand-icon">
            🩺
          </div>

          <div>
            <strong>THYROCARE AI</strong>
            <span>THYROID ANALYSIS</span>
          </div>
        </div>

        <button
          className="mobile-menu-button"
          onClick={() =>
            setMenuOpen((previous) => !previous)
          }
        >
          ☰
        </button>

        <div
          className={`nav-links ${
            menuOpen ? "open" : ""
          }`}
        >
          <button
            className={page === "home" ? "active" : ""}
            onClick={() => goTo("home")}
          >
            Home
          </button>

          <button
            className={
              page === "prediction" ? "active" : ""
            }
            onClick={() => goTo("prediction")}
          >
            Analyze
          </button>

          <button
            className={
              page === "results" ? "active" : ""
            }
            onClick={() => goTo("results")}
          >
            Results
          </button>

          <button
            className={
              page === "about" ? "active" : ""
            }
            onClick={() => goTo("about")}
          >
            About
          </button>

          <button
            className={
              page === "admin" ||
              page === "admin-dashboard"
                ? "active admin-nav-button"
                : "admin-nav-button"
            }
            onClick={() => goTo("admin")}
          >
            🔐 Admin
          </button>
        </div>
      </nav>
    );
  };

  const renderHomePage = () => {
    return (
      <>
        <section className="hero">
          <div className="hero-content">
            <div className="hero-badge">
              ✨ AI POWERED THYROID ANALYSIS
            </div>

            <h1>
              Intelligent Thyroid
              <br />
              <span>Diagnosis & Explainability</span>
            </h1>

            <p>
              Analyze thyroid-related clinical
              measurements using Machine Learning,
              SHAP Explainable AI and Counterfactual
              explanations.
            </p>

            <div className="hero-actions">
              <button
                className="primary-button"
                onClick={() =>
                  goTo("prediction")
                }
              >
                Start Analysis →
              </button>

              <button
                className="secondary-button"
                onClick={() => goTo("about")}
              >
                Explore Project
              </button>
            </div>
          </div>

          <div className="hero-visual">
            <div className="ai-orbit">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="orbit orbit-three" />

              <div className="ai-core">
                <span>AI</span>
                <small>THYROID</small>
              </div>
            </div>
          </div>
        </section>

        <section className="feature-section">
          <div className="section-heading">
            <span>POWERFUL FEATURES</span>

            <h2>
              From Prediction to Explanation
            </h2>

            <p>
              A complete machine learning workflow
              designed for transparent thyroid
              disease analysis.
            </p>
          </div>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon">
                🧠
              </div>

              <h3>Machine Learning</h3>

              <p>
                XGBoost analyzes clinical features
                and generates a thyroid disease
                prediction.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                🔍
              </div>

              <h3>SHAP Explainability</h3>

              <p>
                Understand which clinical features
                contributed to the prediction.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                🔄
              </div>

              <h3>Counterfactual AI</h3>

              <p>
                Explore how changes in important
                measurements can influence the
                prediction.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                📊
              </div>

              <h3>Prediction Report</h3>

              <p>
                View probabilities, explanations
                and generate a downloadable report.
              </p>
            </div>
          </div>
        </section>
      </>
    );
  };

  const renderPredictionPage = () => {
    return (
      <section className="prediction-page">
        <div className="page-heading">
          <span>AI THYROID ANALYSIS</span>

          <h1>Enter Clinical Parameters</h1>

          <p>
            Provide the patient's clinical values
            below to generate an AI prediction.
          </p>
        </div>

        {error && (
          <div className="error-box">
            ⚠️ {error}
          </div>
        )}

        <div className="form-card">
          <div className="form-section">
            <h2>Patient Information</h2>

            <div className="form-grid">
              <div className="input-group">
                <label>Age</label>

                <input
                  type="number"
                  value={form.age}
                  onChange={(e) =>
                    updateField(
                      "age",
                      e.target.value
                    )
                  }
                />
              </div>

              <div className="input-group">
                <label>Sex</label>

                <select
                  value={form.sex}
                  onChange={(e) =>
                    updateField(
                      "sex",
                      Number(e.target.value)
                    )
                  }
                >
                  <option value={0}>
                    Female
                  </option>

                  <option value={1}>
                    Male
                  </option>
                </select>
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2>Thyroid Measurements</h2>

            <div className="form-grid">
              {[
                ["TSH", "TSH"],
                ["TT4", "TT4"],
                ["T4U", "T4U"],
                ["FTI", "FTI"],
              ].map(([label, key]) => (
                <div
                  className="input-group"
                  key={key}
                >
                  <label>{label}</label>

                  <input
                    type="number"
                    step="any"
                    value={form[key]}
                    onChange={(e) =>
                      updateField(
                        key,
                        e.target.value
                      )
                    }
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="form-section">
            <h2>Clinical Indicators</h2>

            <div className="binary-grid">
              {binaryFields
                .filter(
                  (field) =>
                    field !== "sex"
                )
                .map((field) => (
                  <label
                    className="toggle-field"
                    key={field}
                  >
                    <span>
                      {field}
                    </span>

                    <select
                      value={form[field]}
                      onChange={(e) =>
                        updateField(
                          field,
                          Number(
                            e.target.value
                          )
                        )
                      }
                    >
                      <option value={0}>
                        No
                      </option>

                      <option value={1}>
                        Yes
                      </option>
                    </select>
                  </label>
                ))}
            </div>
          </div>

          <div className="form-actions">
            <button
              className="secondary-button"
              onClick={resetForm}
              disabled={loading}
            >
              Reset
            </button>

            <button
              className="primary-button analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : "Analyze Thyroid →"}
            </button>
          </div>
        </div>
      </section>
    );
  };

  const renderResultsPage = () => {
    if (!result) {
      return (
        <section className="empty-results">
          <div className="empty-icon">
            📊
          </div>

          <h1>No Analysis Yet</h1>

          <p>
            Run a thyroid analysis to view the
            prediction and AI explanations.
          </p>

          <button
            className="primary-button"
            onClick={() =>
              goTo("prediction")
            }
          >
            Start Analysis →
          </button>
        </section>
      );
    }

    return (
      <section className="results-page">
        <div className="page-heading">
          <span>ANALYSIS COMPLETE</span>

          <h1>AI Prediction Results</h1>

          <p>
            Machine learning prediction with
            explainable AI insights.
          </p>
        </div>

        <div
          className={`prediction-banner ${
            predictionClass === 1
              ? "positive"
              : "negative"
          }`}
        >
          <div className="prediction-symbol">
            {predictionClass === 1
              ? "⚠️"
              : "✓"}
          </div>

          <div>
            <span>MODEL PREDICTION</span>

            <h2>{predictionLabel}</h2>

            <p>
              Model:{" "}
              {result.model ||
                result.model_name ||
                "XGBoost"}
            </p>
          </div>
        </div>

        <div className="result-grid">
          <div className="result-card">
            <div className="result-card-header">
              <span>
                CLASS 0
              </span>

              <strong>
                {formatPercent(class0)}
              </strong>
            </div>

            <div className="progress-bar">
              <div
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(
                      100,
                      Number(class0) *
                        100
                    )
                  )}%`,
                }}
              />
            </div>

            <p>
              Thyroid Disease Not Predicted
            </p>
          </div>

          <div className="result-card">
            <div className="result-card-header">
              <span>
                CLASS 1
              </span>

              <strong>
                {formatPercent(class1)}
              </strong>
            </div>

            <div className="progress-bar">
              <div
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(
                      100,
                      Number(class1) *
                        100
                    )
                  )}%`,
                }}
              />
            </div>

            <p>
              Thyroid Disease Predicted
            </p>
          </div>
        </div>

        <div className="explanation-section">
          <div className="section-heading left">
            <span>EXPLAINABLE AI</span>

            <h2>
              SHAP Feature Impact
            </h2>

            <p>
              These values show how individual
              features influenced the model
              prediction.
            </p>
          </div>

          {shapData.length > 0 ? (
            <div className="shap-list">
              {shapData
                .slice(0, 10)
                .map((item, index) => (
                  <div
                    className="shap-row"
                    key={`${item.feature}-${index}`}
                  >
                    <div className="shap-name">
                      {item.feature}
                    </div>

                    <div className="shap-value">
                      {formatNumber(
                        item.value
                      )}
                    </div>

                    <div className="shap-direction">
                      {item.value > 0
                        ? "↑"
                        : item.value < 0
                        ? "↓"
                        : "→"}
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <div className="empty-card">
              SHAP explanation is not available.
            </div>
          )}
        </div>

        <div className="explanation-section">
          <div className="section-heading left">
            <span>COUNTERFACTUAL AI</span>

            <h2>
              What Could Change the Prediction?
            </h2>

            <p>
              Counterfactual examples demonstrate
              how changing important input values
              may affect the model prediction.
            </p>
          </div>

          {counterfactualData.length > 0 ? (
            <div className="counterfactual-grid">
              {counterfactualData.map(
                (item, index) => (
                  <div
                    className="counterfactual-card"
                    key={index}
                  >
                    <span>
                      {item.feature}
                    </span>

                    <div className="cf-values">
                      <strong>
                        {item.from}
                      </strong>

                      <span>→</span>

                      <strong>
                        {item.to}
                      </strong>
                    </div>

                    <small>
                      Prediction:{" "}
                      {item.prediction}
                    </small>

                    {item.probability !==
                      null && (
                      <small>
                        Probability:{" "}
                        {formatPercent(
                          item.probability
                        )}
                      </small>
                    )}
                  </div>
                )
              )}
            </div>
          ) : (
            <div className="empty-card">
              No counterfactual examples available.
            </div>
          )}
        </div>

        <div className="result-actions">
          <button
            className="secondary-button"
            onClick={() =>
              goTo("prediction")
            }
          >
            ← New Analysis
          </button>

          <button
            className="primary-button"
            onClick={() => {
              const report = `
THYROCARE AI
THYROID ANALYSIS REPORT
--------------------------------

Prediction:
${predictionLabel}

Class 0 Probability:
${formatPercent(class0)}

Class 1 Probability:
${formatPercent(class1)}

Model:
${result.model || result.model_name || "XGBoost"}

SHAP FEATURES
--------------------------------
${shapData
  .slice(0, 10)
  .map(
    (item) =>
      `${item.feature}: ${formatNumber(
        item.value
      )}`
  )
  .join("\n")}

COUNTERFACTUALS
--------------------------------
${counterfactualData
  .map(
    (item) =>
      `${item.feature}: ${item.from} -> ${item.to}`
  )
  .join("\n")}

--------------------------------
ThyroCare AI
Final Year B.Tech Project
Machine Learning & Explainable AI
`;

              const blob =
                new Blob([report], {
                  type: "text/plain",
                });

              const url =
                URL.createObjectURL(blob);

              const link =
                document.createElement("a");

              link.href = url;
              link.download =
                "thyrocare-ai-report.txt";

              link.click();

              URL.revokeObjectURL(url);
            }}
          >
            Download Report ↓
          </button>
        </div>
      </section>
    );
  };

  const renderAboutPage = () => {
    return (
      <section className="about-page">
        <div className="page-heading">
          <span>ABOUT THE PROJECT</span>

          <h1>
            ThyroCare AI
          </h1>

          <p>
            Enhancing Thyroid Disease Diagnosis
            With Machine Learning and
            Counterfactual Explainable AI.
          </p>
        </div>

        <div className="about-grid">
          <div className="about-card">
            <div className="feature-icon">
              🤖
            </div>

            <h2>Machine Learning</h2>

            <p>
              The system uses an XGBoost machine
              learning model trained on thyroid
              clinical data.
            </p>
          </div>

          <div className="about-card">
            <div className="feature-icon">
              🔎
            </div>

            <h2>Explainable AI</h2>

            <p>
              SHAP values provide insight into
              which clinical features influence
              the model's output.
            </p>
          </div>

          <div className="about-card">
            <div className="feature-icon">
              🔄
            </div>

            <h2>Counterfactual Analysis</h2>

            <p>
              Counterfactual examples show how
              changing selected measurements can
              affect predictions.
            </p>
          </div>
        </div>

        <div className="technology-card">
          <h2>Technology Stack</h2>

          <div className="technology-list">
            <span>React</span>
            <span>Vite</span>
            <span>FastAPI</span>
            <span>Python</span>
            <span>XGBoost</span>
            <span>SHAP</span>
            <span>Explainable AI</span>
            <span>Render</span>
            <span>Vercel</span>
          </div>
        </div>
      </section>
    );
  };

  const renderAdminDashboard = () => {
    return (
      <section className="admin-dashboard">
        <div className="admin-dashboard-header">
          <div>
            <span>
              ADMINISTRATION
            </span>

            <h1>
              ThyroCare AI Dashboard
            </h1>

            <p>
              Manage datasets, machine learning
              experiments and prediction analytics.
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={() => {
              setAdminAuthenticated(false);
              goTo("home");
            }}
          >
            Logout
          </button>
        </div>

        <div className="admin-stats">
          <div className="admin-stat-card">
            <span>MODEL</span>

            <strong>
              XGBoost
            </strong>

            <small>
              Production Model
            </small>
          </div>

          <div className="admin-stat-card">
            <span>FEATURES</span>

            <strong>
              25
            </strong>

            <small>
              Clinical Features
            </small>
          </div>

          <div className="admin-stat-card">
            <span>EXPLAINABILITY</span>

            <strong>
              SHAP
            </strong>

            <small>
              Enabled
            </small>
          </div>

          <div className="admin-stat-card">
            <span>COUNTERFACTUAL</span>

            <strong>
              ACTIVE
            </strong>

            <small>
              AI Explanation
            </small>
          </div>
        </div>

        <div className="admin-module-grid">
          <div className="admin-module">
            <div className="admin-module-icon">
              📁
            </div>

            <h2>
              Dataset Management
            </h2>

            <p>
              Upload and inspect thyroid datasets.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>

          <div className="admin-module">
            <div className="admin-module-icon">
              🧹
            </div>

            <h2>
              Dataset Preprocessing
            </h2>

            <p>
              Remove duplicates, inspect missing
              values and prepare training data.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>

          <div className="admin-module">
            <div className="admin-module-icon">
              🧠
            </div>

            <h2>
              Algorithm Training
            </h2>

            <p>
              Train machine learning algorithms
              using the uploaded dataset.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>

          <div className="admin-module">
            <div className="admin-module-icon">
              🎯
            </div>

            <h2>
              Test Accuracy
            </h2>

            <p>
              View accuracy, precision, recall,
              F1-score and ROC-AUC.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>

          <div className="admin-module">
            <div className="admin-module-icon">
              📊
            </div>

            <h2>
              Algorithm Comparison
            </h2>

            <p>
              Compare Logistic Regression,
              Decision Tree, Random Forest,
              SVM and XGBoost.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>

          <div className="admin-module">
            <div className="admin-module-icon">
              🗃️
            </div>

            <h2>
              Prediction History
            </h2>

            <p>
              Review previous prediction requests
              and model results.
            </p>

            <button disabled>
              Coming Soon
            </button>
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="app">
      {page !== "admin" &&
        page !== "admin-dashboard" &&
        renderNavbar()}

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
            setAdminAuthenticated(true);
            setPage("admin-dashboard");
          }}
          onBack={() => goTo("home")}
        />
      )}

      {page === "admin-dashboard" &&
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
          Final Year B.Tech Project • Machine
          Learning & Explainable AI
        </p>
      </footer>
    </div>
  );
    }
