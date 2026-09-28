import React, { useState } from "react";
import axios from "axios";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Download,
  HeartPulse,
  Info,
  Menu,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  X,
} from "lucide-react";

const API_URL =
  "https://thyrocare-diagnosis-ai.onrender.com";

/* =====================================================
   DEFAULT FORM
===================================================== */

const defaultForm = {
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

/* =====================================================
   FORM FIELDS
===================================================== */

const numericFields = [
  ["age", "Age", 1, 120, 1],
  ["TSH", "TSH", 0, 100, 0.01],
  ["TT4", "TT4", 0, 1000, 0.1],
  ["T4U", "T4U", 0, 10, 0.01],
  ["FTI", "FTI", 0, 1000, 0.1],
];

const binaryFields = [
  ["sex", "Sex"],
  ["on thyroxine", "On thyroxine"],
  ["query on thyroxine", "Query on thyroxine"],
  [
    "on antithyroid medication",
    "On antithyroid medication",
  ],
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

/* =====================================================
   HELPERS
===================================================== */

function toNumber(value, fallback = 0) {
  const n = Number(value);

  return Number.isFinite(n) ? n : fallback;
}

function toPercent(value) {
  let n = toNumber(value);

  if (n >= 0 && n <= 1) {
    n *= 100;
  }

  return Math.max(0, Math.min(100, n));
}

function percentText(value) {
  return `${toPercent(value).toFixed(2)}%`;
}

/* =====================================================
   OBJECT -> READABLE TEXT
===================================================== */

function readable(value) {
  if (value === null || value === undefined) {
    return "";
  }

  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => readable(item))
      .filter(Boolean)
      .join(" • ");
  }

  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, val]) => {
        const converted = readable(val);

        return converted
          ? `${key}: ${converted}`
          : key;
      })
      .join(" • ");
  }

  return String(value);
}

/* =====================================================
   READ PREDICTION
===================================================== */

function getPrediction(data) {
  const values = [
    data?.prediction,
    data?.predicted_class,
    data?.predictedClass,
    data?.class,
  ];

  for (const value of values) {
    if (typeof value === "number") {
      return value === 1 ? 1 : 0;
    }

    if (typeof value === "string") {
      const text = value.toLowerCase();

      if (
        text === "1" ||
        text.includes("disease predicted")
      ) {
        return 1;
      }

      if (
        text === "0" ||
        text.includes("not predicted")
      ) {
        return 0;
      }
    }
  }

  return 0;
}

/* =====================================================
   READ MODEL
===================================================== */

function getModel(data) {
  if (typeof data?.model === "string") {
    return data.model;
  }

  if (data?.model?.name) {
    return String(data.model.name);
  }

  if (data?.model?.model_name) {
    return String(data.model.model_name);
  }

  if (data?.model_name) {
    return String(data.model_name);
  }

  return "XGBoost";
}

/* =====================================================
   READ PROBABILITY
===================================================== */

function getProbability(data, classNumber) {
  const candidates =
    classNumber === 0
      ? [
          data?.probability_class_0,
          data?.class_0_probability,
          data?.probabilities?.class_0,
          data?.probabilities?.["0"],
          data?.probability?.class_0,
          data?.probability?.["0"],
        ]
      : [
          data?.probability_class_1,
          data?.class_1_probability,
          data?.probabilities?.class_1,
          data?.probabilities?.["1"],
          data?.probability?.class_1,
          data?.probability?.["1"],
        ];

  for (const value of candidates) {
    if (value !== undefined && value !== null) {
      return toNumber(value);
    }
  }

  if (Array.isArray(data?.probabilities)) {
    return toNumber(data.probabilities[classNumber]);
  }

  if (Array.isArray(data?.probability)) {
    return toNumber(data.probability[classNumber]);
  }

  return 0;
}

/* =====================================================
   SHAP
===================================================== */

function normalizeShap(data) {
  const raw =
    data?.shap ??
    data?.shap_values ??
    data?.explanation ??
    data?.feature_importance ??
    [];

  if (Array.isArray(raw)) {
    return raw
      .map((item, index) => {
        if (typeof item === "number") {
          return {
            feature: `Feature ${index + 1}`,
            value: item,
          };
        }

        if (
          typeof item === "object" &&
          item !== null
        ) {
          const feature =
            item.feature ??
            item.name ??
            item.feature_name ??
            item.column ??
            `Feature ${index + 1}`;

          const value =
            item.value ??
            item.shap_value ??
            item.impact ??
            item.contribution ??
            item.mean_abs_shap ??
            0;

          return {
            feature: readable(feature),
            value: toNumber(value),
          };
        }

        return null;
      })
      .filter(Boolean)
      .sort(
        (a, b) =>
          Math.abs(b.value) -
          Math.abs(a.value)
      )
      .slice(0, 10);
  }

  if (
    typeof raw === "object" &&
    raw !== null
  ) {
    return Object.entries(raw)
      .map(([feature, value]) => {
        let actualValue = value;

        if (
          typeof value === "object" &&
          value !== null
        ) {
          actualValue =
            value.value ??
            value.impact ??
            value.shap_value ??
            value.contribution ??
            0;
        }

        return {
          feature: String(feature),
          value: toNumber(actualValue),
        };
      })
      .sort(
        (a, b) =>
          Math.abs(b.value) -
          Math.abs(a.value)
      )
      .slice(0, 10);
  }

  return [];
}

/* =====================================================
   COUNTERFACTUALS
===================================================== */

function normalizeCounterfactuals(data) {
  const raw =
    data?.counterfactuals ??
    data?.counterfactual ??
    data?.counterfactual_explanations ??
    [];

  if (Array.isArray(raw)) {
    return raw
      .map((item) => readable(item))
      .filter(Boolean);
  }

  if (
    typeof raw === "object" &&
    raw !== null
  ) {
    return Object.entries(raw)
      .map(
        ([key, value]) =>
          `${key}: ${readable(value)}`
      )
      .filter(Boolean);
  }

  if (raw) {
    return [readable(raw)];
  }

  return [];
}

/* =====================================================
   APP
===================================================== */

export default function App() {
  const [page, setPage] = useState("home");
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  /* ===================================================
     NAVIGATION
  =================================================== */

  function navigate(target) {
    setPage(target);
    setMenuOpen(false);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  /* ===================================================
     UPDATE FIELD
  =================================================== */

  function updateField(key, value) {
    setForm((previous) => ({
      ...previous,
      [key]: value,
    }));
  }

  /* ===================================================
     PREDICTION
  =================================================== */

  async function handlePrediction(event) {
    event.preventDefault();

    setLoading(true);
    setError("");

    try {
      /*
       * IMPORTANT:
       * The frontend uses readable names with spaces.
       * FastAPI expects underscore names.
       */

      const payload = {
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

      console.log(
        "Sending prediction payload:",
        payload
      );

      const response = await axios.post(
        `${API_URL}/predict`,
        payload,
        {
          headers: {
            "Content-Type":
              "application/json",
          },
          timeout: 60000,
        }
      );

      console.log(
        "Backend response:",
        response.data
      );

      const data = response.data || {};

      const prediction =
        getPrediction(data);

      const class0 =
        getProbability(data, 0);

      const class1 =
        getProbability(data, 1);

      const normalizedResult = {
        prediction,

        prediction_label:
          prediction === 1
            ? "Thyroid Disease Predicted"
            : "Thyroid Disease Not Predicted",

        model: getModel(data),

        probabilities: {
          class_0: class0,
          class_1: class1,
        },

        shap: normalizeShap(data),

        counterfactuals:
          normalizeCounterfactuals(data),

        raw: data,
      };

      console.log(
        "Final result:",
        normalizedResult
      );

      setResult(normalizedResult);
      setPage("results");

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (err) {
      console.error(
        "Prediction error:",
        err
      );

      if (err.response?.data?.detail) {
        setError(
          readable(
            err.response.data.detail
          )
        );
      } else if (
        err.response?.data?.message
      ) {
        setError(
          readable(
            err.response.data.message
          )
        );
      } else if (
        err.code === "ECONNABORTED"
      ) {
        setError(
          "The prediction server took too long to respond."
        );
      } else {
        setError(
          "Unable to connect to prediction server."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  /* ===================================================
     DOWNLOAD REPORT
  =================================================== */

  function downloadReport() {
    if (!result) return;

    const shapText =
      result.shap.length > 0
        ? result.shap
            .map(
              (item) =>
                `${item.feature}: ${item.value.toFixed(
                  6
                )}`
            )
            .join("\n")
        : "SHAP data not available.";

    const cfText =
      result.counterfactuals.length > 0
        ? result.counterfactuals.join("\n")
        : "Counterfactual data not available.";

    const report = `
THYROCARE AI
========================================

MODEL PREDICTION
${result.prediction_label}

MODEL
${result.model}

CLASS 0 PROBABILITY
${percentText(
  result.probabilities.class_0
)}

CLASS 1 PROBABILITY
${percentText(
  result.probabilities.class_1
)}

========================================
SHAP EXPLANATION
========================================

${shapText}

========================================
COUNTERFACTUAL EXPLANATION
========================================

${cfText}

========================================
DISCLAIMER
========================================

This application is intended for
educational and research purposes.
It is not a substitute for professional
medical diagnosis.
`;

    const blob = new Blob([report], {
      type: "text/plain",
    });

    const url =
      URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;
    link.download =
      "thyrocare-ai-report.txt";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  }

  /* ===================================================
     NAVBAR
  =================================================== */

  function Navbar() {
    return (
      <header className="navbar">
        <button
          className="brand"
          onClick={() =>
            navigate("home")
          }
        >
          <span className="brand-icon">
            <HeartPulse size={24} />
          </span>

          <span>
            <strong>ThyroCare</strong>
            <small>AI DIAGNOSIS</small>
          </span>
        </button>

        <nav
          className={`nav-links ${
            menuOpen
              ? "mobile-open"
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
              navigate("home")
            }
          >
            Home
          </button>

          <button
            className={
              page === "predict"
                ? "active"
                : ""
            }
            onClick={() =>
              navigate("predict")
            }
          >
            Prediction
          </button>

          <button
            className={
              page === "about"
                ? "active"
                : ""
            }
            onClick={() =>
              navigate("about")
            }
          >
            About
          </button>
        </nav>

        <button
          className="nav-cta"
          onClick={() =>
            navigate("predict")
          }
        >
          Start Test
          <ArrowRight size={17} />
        </button>

        <button
          className="mobile-menu-button"
          onClick={() =>
            setMenuOpen(
              !menuOpen
            )
          }
        >
          {menuOpen ? (
            <X size={23} />
          ) : (
            <Menu size={23} />
          )}
        </button>
      </header>
    );
  }

  /* ===================================================
     HOME
  =================================================== */

  function Home() {
    return (
      <main>
        <section className="hero-section">
          <div className="hero-content">
            <div className="eyebrow">
              <Sparkles size={15} />
              AI-POWERED THYROID ANALYSIS
            </div>

            <h1>
              Understand your thyroid
              <span>
                with intelligent AI.
              </span>
            </h1>

            <p className="hero-description">
              ThyroCare combines machine
              learning with explainable AI
              to provide thyroid disease
              predictions and understandable
              insights.
            </p>

            <div className="hero-actions">
              <button
                className="primary-button"
                onClick={() =>
                  navigate("predict")
                }
              >
                Start Prediction
                <ArrowRight size={20} />
              </button>

              <button
                className="secondary-button"
                onClick={() =>
                  navigate("about")
                }
              >
                Learn About AI
                <ChevronRight size={20} />
              </button>
            </div>

            <div className="trust-row">
              <div>
                <Brain size={20} />
                Machine Learning
              </div>

              <div>
                <ShieldCheck size={20} />
                Explainable AI
              </div>

              <div>
                <Activity size={20} />
                Instant Analysis
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-glow glow-one" />
            <div className="hero-glow glow-two" />

            <div className="floating-card floating-one">
              <Brain size={25} />

              <span>
                <strong>XGBoost</strong>
                <small>
                  Prediction model
                </small>
              </span>
            </div>

            <div className="floating-card floating-two">
              <Sparkles size={25} />

              <span>
                <strong>SHAP</strong>
                <small>
                  Explainable result
                </small>
              </span>
            </div>

            <div className="medical-card">
              <div className="card-top">
                <span className="status-dot" />

                AI DIAGNOSTIC ENGINE

                <Activity size={22} />
              </div>

              <div className="heart-visual">
                <div className="pulse-ring ring-one" />
                <div className="pulse-ring ring-two" />

                <div className="heart-center">
                  <HeartPulse size={55} />
                </div>
              </div>

              <div className="analysis-line">
                <div>
                  <small>
                    MODEL STATUS
                  </small>

                  <strong>
                    Ready for analysis
                  </strong>
                </div>

                <CheckCircle2
                  size={25}
                />
              </div>
            </div>
          </div>
        </section>
      </main>
    );
  }

  /* ===================================================
     PREDICTION
  =================================================== */

  function Prediction() {
    return (
      <main className="page-section">
        <button
          className="back-button"
          onClick={() =>
            navigate("home")
          }
        >
          <ArrowLeft size={17} />
          Back to home
        </button>

        <div className="page-heading">
          <div className="eyebrow">
            <Stethoscope size={15} />
            THYROID PREDICTION
          </div>

          <h2>
            Enter clinical information
          </h2>

          <p>
            Enter the available clinical
            information below. The trained
            machine learning model will
            analyze the values.
          </p>
        </div>

        <form
          className="prediction-form"
          onSubmit={
            handlePrediction
          }
        >
          <div className="form-card">
            <div className="form-card-heading">
              <div className="section-icon">
                <Activity size={22} />
              </div>

              <div>
                <h3>
                  Clinical measurements
                </h3>

                <p>
                  Enter the patient's
                  measurements and laboratory
                  values.
                </p>
              </div>
            </div>

            <div className="form-grid">
              {numericFields.map(
                ([
                  key,
                  label,
                  min,
                  max,
                  step,
                ]) => (
                  <label
                    className="input-group"
                    key={key}
                  >
                    <span>{label}</span>

                    <input
                      type="number"
                      min={min}
                      max={max}
                      step={step}
                      value={form[key]}
                      onChange={(e) =>
                        updateField(
                          key,
                          e.target.value
                        )
                      }
                    />
                  </label>
                )
              )}
            </div>
          </div>

          <div className="form-card">
            <div className="form-card-heading">
              <div className="section-icon">
                <ShieldCheck
                  size={22}
                />
              </div>

              <div>
                <h3>
                  Clinical indicators
                </h3>

                <p>
                  Select Yes or No for each
                  indicator.
                </p>
              </div>
            </div>

            <div className="indicator-grid">
              {binaryFields.map(
                ([key, label]) => (
                  <label
                    className="toggle-row"
                    key={key}
                  >
                    <span>{label}</span>

                    <select
                      value={form[key]}
                      onChange={(e) =>
                        updateField(
                          key,
                          Number(
                            e.target.value
                          )
                        )
                      }
                    >
                      <option value={0}>
                        No (0)
                      </option>

                      <option value={1}>
                        Yes (1)
                      </option>
                    </select>
                  </label>
                )
              )}
            </div>
          </div>

          {error && (
            <div className="error-box">
              <CircleAlert size={20} />
              <span>{error}</span>
            </div>
          )}

          <button
            className="predict-button"
            type="submit"
            disabled={loading}
          >
            {loading ? (
              <>
                <Activity
                  className="spin"
                  size={20}
                />

                Analyzing...
              </>
            ) : (
              <>
                Analyze with AI
                <ArrowRight size={20} />
              </>
            )}
          </button>

          <div className="medical-disclaimer">
            <Info size={18} />

            <span>
              This tool is intended for
              educational and research
              purposes and should not replace
              professional medical advice.
            </span>
          </div>
        </form>
      </main>
    );
  }

  /* ===================================================
     RESULTS
  =================================================== */

  function Results() {
    if (!result) {
      return (
        <main className="page-section">
          <div className="page-heading">
            <div className="eyebrow">
              <Info size={15} />
              NO RESULT
            </div>

            <h2>
              No prediction available
            </h2>

            <p>
              Run a prediction first to
              view the result.
            </p>

            <button
              className="primary-button"
              onClick={() =>
                navigate("predict")
              }
            >
              Start Prediction
              <ArrowRight size={19} />
            </button>
          </div>
        </main>
      );
    }

    const positive =
      result.prediction === 1;

    return (
      <main className="page-section">
        <button
          className="back-button"
          onClick={() =>
            navigate("predict")
          }
        >
          <ArrowLeft size={17} />
          New prediction
        </button>

        <div className="page-heading">
          <div className="eyebrow">
            <Sparkles size={15} />
            YOUR PREDICTION RESULT
          </div>

          <h2>
            AI analysis completed
          </h2>

          <p>
            The machine learning model has
            analyzed the submitted clinical
            information.
          </p>
        </div>

        <div
          className={`result-banner ${
            positive
              ? "positive"
              : "negative"
          }`}
        >
          <div className="result-icon">
            {positive ? (
              <CircleAlert
                size={34}
              />
            ) : (
              <CheckCircle2
                size={34}
              />
            )}
          </div>

          <div>
            <span>
              MODEL PREDICTION
            </span>

            <h2>
              {result.prediction_label}
            </h2>

            <p>
              Model:{" "}
              <strong>
                {result.model}
              </strong>
            </p>
          </div>
        </div>

        <div className="results-grid">
          <section className="result-card">
            <div className="result-card-title">
              <Activity size={19} />
              Class probabilities
            </div>

            <div className="probability">
              <div className="probability-head">
                <span>
                  Class 0
                </span>

                <strong>
                  {percentText(
                    result
                      .probabilities
                      .class_0
                  )}
                </strong>
              </div>

              <div className="progress">
                <div
                  className="progress-bar"
                  style={{
                    width: `${toPercent(
                      result
                        .probabilities
                        .class_0
                    )}%`,
                  }}
                />
              </div>
            </div>

            <div className="probability">
              <div className="probability-head">
                <span>
                  Class 1
                </span>

                <strong>
                  {percentText(
                    result
                      .probabilities
                      .class_1
                  )}
                </strong>
              </div>

              <div className="progress">
                <div
                  className="progress-bar"
                  style={{
                    width: `${toPercent(
                      result
                        .probabilities
                        .class_1
                    )}%`,
                  }}
                />
              </div>
            </div>
          </section>

          <section className="result-card">
            <div className="result-card-title">
              <Sparkles size={19} />
              Explainable AI
            </div>

            <p className="result-text">
              SHAP explains which input
              features contributed to the
              model prediction.
            </p>

            <div className="shap-list">
              {result.shap.length >
              0 ? (
                result.shap.map(
                  (
                    item,
                    index
                  ) => (
                    <div
                      className="shap-row"
                      key={index}
                    >
                      <span>
                        {readable(
                          item.feature
                        )}
                      </span>

                      <strong>
                        {toNumber(
                          item.value
                        ).toFixed(4)}
                      </strong>
                    </div>
                  )
                )
              ) : (
                <div className="empty-small">
                  SHAP explanation was
                  not returned by the
                  backend.
                </div>
              )}
            </div>
          </section>
        </div>

        <section className="result-card wide-card">
          <div className="result-card-title">
            <Brain size={19} />
            Counterfactual explanation
          </div>

          <p className="result-text">
            Counterfactual explanations
            show changes that could
            potentially change the model
            prediction.
          </p>

          <div className="counterfactual-list">
            {result
              .counterfactuals
              .length > 0 ? (
              result.counterfactuals.map(
                (
                  item,
                  index
                ) => (
                  <div
                    className="counterfactual-item"
                    key={index}
                  >
                    {readable(item)}
                  </div>
                )
              )
            ) : (
              <div className="empty-small">
                Counterfactual explanation
                was not returned by the
                backend.
              </div>
            )}
          </div>
        </section>

        <div className="result-actions">
          <button
            className="primary-button"
            onClick={
              downloadReport
            }
          >
            <Download size={18} />
            Download Report
          </button>
        </div>

        <div className="medical-disclaimer">
          <Info size={18} />

          <span>
            This prediction is generated
            by a machine learning model for
            educational and research
            purposes.
          </span>
        </div>
      </main>
    );
  }

  /* ===================================================
     ABOUT
  =================================================== */

  function About() {
    return (
      <main className="page-section">
        <div className="page-heading">
          <div className="eyebrow">
            <Sparkles size={15} />
            ABOUT THYROCARE
          </div>

          <h2>
            AI with understandable
            results.
          </h2>

          <p>
            ThyroCare combines machine
            learning prediction with
            explainable artificial
            intelligence.
          </p>
        </div>

        <div className="about-grid">
          <section className="about-card">
            <Brain size={30} />

            <h3>
              Machine Learning
            </h3>

            <p>
              Uses a trained XGBoost model
              to analyze thyroid-related
              clinical features.
            </p>
          </section>

          <section className="about-card">
            <Sparkles size={30} />

            <h3>
              Explainable AI
            </h3>

            <p>
              SHAP explanations help
              identify features that
              influence the model output.
            </p>
          </section>

          <section className="about-card">
            <ShieldCheck size={30} />

            <h3>
              Transparent Results
            </h3>

            <p>
              Displays prediction classes
              and probability information.
            </p>
          </section>

          <section className="about-card">
            <HeartPulse size={30} />

            <h3>
              Academic Project
            </h3>

            <p>
              Designed as a machine
              learning and explainable AI
              project.
            </p>
          </section>
        </div>
      </main>
    );
  }

  /* ===================================================
     FOOTER
  =================================================== */

  function Footer() {
    return (
      <footer className="footer">
        <div className="footer-brand">
          <span className="brand-icon">
            <HeartPulse size={20} />
          </span>

          <div>
            <strong>
              ThyroCare AI
            </strong>

            <p>
              Intelligent thyroid analysis.
            </p>
          </div>
        </div>

        <div className="footer-note">
          <ShieldCheck size={15} />
          Educational and research
          purposes
        </div>
      </footer>
    );
  }

  /* ===================================================
     MAIN
  =================================================== */

  return (
    <div className="app-shell">
      <Navbar />

      {page === "home" && <Home />}

      {page === "predict" && (
        <Prediction />
      )}

      {page === "results" && (
        <Results />
      )}

      {page === "about" && (
        <About />
      )}

      <Footer />
    </div>
  );
        }
