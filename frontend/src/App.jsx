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

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

/* =========================================================
   DEFAULT FORM VALUES
========================================================= */

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
  T4U: 1.0,
  "FTI measured": 1,
  FTI: 110,
};

/* =========================================================
   FIELD DEFINITIONS
========================================================= */

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

/* =========================================================
   HELPERS
========================================================= */

function safeNumber(value, fallback = 0) {
  const number = Number(value);

  if (Number.isFinite(number)) {
    return number;
  }

  return fallback;
}

function percentage(value) {
  let number = safeNumber(value, 0);

  /*
    Backend may return:
    0.998
    OR
    99.8

    Convert both to a percentage.
  */
  if (number >= 0 && number <= 1) {
    number *= 100;
  }

  return Math.max(0, Math.min(100, number));
}

function formatPercentage(value) {
  return `${percentage(value).toFixed(2)}%`;
}

function getPrediction(data) {
  const possibleValues = [
    data?.prediction,
    data?.predicted_class,
    data?.class,
    data?.result,
  ];

  for (const value of possibleValues) {
    if (typeof value === "number") {
      return value;
    }

    if (typeof value === "string") {
      if (value === "1") return 1;
      if (value === "0") return 0;

      if (value.toLowerCase().includes("predicted")) {
        return 1;
      }

      if (value.toLowerCase().includes("not predicted")) {
        return 0;
      }
    }
  }

  return 0;
}

function getModelName(data) {
  if (!data) return "XGBoost";

  if (typeof data.model === "string") {
    return data.model;
  }

  if (data.model?.name) {
    return String(data.model.name);
  }

  if (data.model?.model_name) {
    return String(data.model.model_name);
  }

  if (data.model_name) {
    return String(data.model_name);
  }

  return "XGBoost";
}

function getClassProbability(data, classNumber) {
  const key = `class_${classNumber}`;

  const directCandidates =
    classNumber === 0
      ? [
          data?.probability_class_0,
          data?.class_0_probability,
          data?.probabilities?.class_0,
          data?.probabilities?.["0"],
        ]
      : [
          data?.probability_class_1,
          data?.class_1_probability,
          data?.probabilities?.class_1,
          data?.probabilities?.["1"],
        ];

  for (const value of directCandidates) {
    if (
      typeof value === "number" ||
      (typeof value === "string" && value.trim() !== "")
    ) {
      return safeNumber(value, 0);
    }
  }

  /*
    Support probability arrays such as:
    [0.998, 0.002]
  */
  if (Array.isArray(data?.probabilities)) {
    return safeNumber(data.probabilities[classNumber], 0);
  }

  /*
    Support model_probability object.
  */
  if (data?.model_probability) {
    if (Array.isArray(data.model_probability)) {
      return safeNumber(data.model_probability[classNumber], 0);
    }

    return safeNumber(
      data.model_probability[key] ??
        data.model_probability[String(classNumber)],
      0
    );
  }

  return 0;
}

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

        if (typeof item === "object" && item !== null) {
          return {
            feature:
              item.feature ??
              item.name ??
              item.feature_name ??
              `Feature ${index + 1}`,
            value: safeNumber(
              item.value ??
                item.shap_value ??
                item.impact ??
                item.contribution,
              0
            ),
          };
        }

        return null;
      })
      .filter(Boolean)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 10);
  }

  if (typeof raw === "object" && raw !== null) {
    return Object.entries(raw)
      .map(([feature, value]) => ({
        feature,
        value: safeNumber(
          typeof value === "object" && value !== null
            ? value.value ?? value.impact ?? value.shap_value
            : value,
          0
        ),
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 10);
  }

  return [];
}

function normalizeCounterfactuals(data) {
  const raw =
    data?.counterfactuals ??
    data?.counterfactual ??
    data?.counterfactual_explanations ??
    [];

  if (Array.isArray(raw)) {
    return raw.map((item) => {
      if (typeof item === "string") {
        return item;
      }

      if (typeof item === "object" && item !== null) {
        if (item.text) return String(item.text);

        if (item.description) {
          return String(item.description);
        }

        try {
          return Object.entries(item)
            .map(([key, value]) => `${key}: ${String(value)}`)
            .join(" • ");
        } catch {
          return "Counterfactual explanation available.";
        }
      }

      return String(item);
    });
  }

  if (typeof raw === "object" && raw !== null) {
    return Object.entries(raw).map(
      ([key, value]) => `${key}: ${String(value)}`
    );
  }

  return [];
}

/* =========================================================
   APP
========================================================= */

export default function App() {
  const [page, setPage] = useState("home");
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);

  /* =======================================================
     NAVIGATION
  ======================================================= */

  const navigate = (target) => {
    setPage(target);
    setMobileMenu(false);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  /* =======================================================
     FORM UPDATE
  ======================================================= */

  const updateField = (key, value) => {
    setForm((previous) => ({
      ...previous,
      [key]: value,
    }));
  };

  /* =======================================================
     PREDICTION
  ======================================================= */

  const handlePrediction = async (event) => {
    event.preventDefault();

    setLoading(true);
    setError("");

    try {
      /*
        Make sure all numeric fields are actually numbers.
      */
      const payload = {};

      Object.entries(form).forEach(([key, value]) => {
        payload[key] = Number(value);
      });

      console.log("Sending prediction payload:", payload);

      const response = await axios.post(
        `${API_URL}/predict`,
        payload,
        {
          headers: {
            "Content-Type": "application/json",
          },
          timeout: 60000,
        }
      );

      console.log("Backend response:", response.data);

      const data = response.data || {};

      const prediction = getPrediction(data);

      const class0 = getClassProbability(data, 0);
      const class1 = getClassProbability(data, 1);

      const modelName = getModelName(data);

      const normalizedResult = {
        raw: data,

        prediction,

        prediction_label:
          prediction === 1
            ? "Thyroid Disease Predicted"
            : "Thyroid Disease Not Predicted",

        model: modelName,

        probabilities: {
          class_0: class0,
          class_1: class1,
        },

        shap: normalizeShap(data),

        counterfactuals: normalizeCounterfactuals(data),
      };

      console.log("Normalized result:", normalizedResult);

      setResult(normalizedResult);
      setPage("results");

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (err) {
      console.error("Prediction error:", err);

      let message =
        "Unable to connect to the prediction server.";

      if (err.response?.data?.detail) {
        message = String(err.response.data.detail);
      } else if (err.response?.data?.message) {
        message = String(err.response.data.message);
      } else if (err.code === "ECONNABORTED") {
        message =
          "The prediction server took too long to respond. Please try again.";
      }

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  /* =======================================================
     DOWNLOAD REPORT
  ======================================================= */

  const downloadReport = () => {
    if (!result) return;

    const report = `
THYROCARE AI
AI-POWERED THYROID ANALYSIS
========================================

Prediction:
${result.prediction_label}

Model:
${result.model}

Class 0 Probability:
${formatPercentage(result.probabilities.class_0)}

Class 1 Probability:
${formatPercentage(result.probabilities.class_1)}

========================================

SHAP EXPLANATION
========================================

${
  result.shap.length
    ? result.shap
        .map(
          (item) =>
            `${item.feature}: ${safeNumber(item.value).toFixed(6)}`
        )
        .join("\n")
    : "SHAP explanation was not returned by the backend."
}

========================================

COUNTERFACTUAL EXPLANATION
========================================

${
  result.counterfactuals.length
    ? result.counterfactuals.join("\n")
    : "Counterfactual explanation was not returned by the backend."
}

========================================

DISCLAIMER
========================================

This application is intended for educational and research purposes.
It is not a substitute for professional medical diagnosis.
`;

    const blob = new Blob([report], {
      type: "text/plain;charset=utf-8",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "thyrocare-ai-report.txt";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
  };

  /* =======================================================
     NAVBAR
  ======================================================= */

  const Navbar = () => (
    <header className="navbar">
      <button
        className="brand"
        onClick={() => navigate("home")}
        aria-label="ThyroCare home"
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
          mobileMenu ? "mobile-open" : ""
        }`}
      >
        <button
          className={page === "home" ? "active" : ""}
          onClick={() => navigate("home")}
        >
          Home
        </button>

        <button
          className={page === "predict" ? "active" : ""}
          onClick={() => navigate("predict")}
        >
          Prediction
        </button>

        <button
          className={page === "about" ? "active" : ""}
          onClick={() => navigate("about")}
        >
          About
        </button>
      </nav>

      <button
        className="nav-cta"
        onClick={() => navigate("predict")}
      >
        Start Test
        <ArrowRight size={17} />
      </button>

      <button
        className="mobile-menu-button"
        onClick={() => setMobileMenu((value) => !value)}
        aria-label="Open menu"
      >
        {mobileMenu ? <X size={23} /> : <Menu size={23} />}
      </button>
    </header>
  );

  /* =======================================================
     HOME
  ======================================================= */

  const HomePage = () => (
    <>
      <section className="hero-section">
        <div className="hero-content">
          <div className="eyebrow">
            <Sparkles size={15} />
            AI-POWERED THYROID ANALYSIS
          </div>

          <h1>
            Understand your thyroid
            <span>with intelligent AI.</span>
          </h1>

          <p className="hero-description">
            ThyroCare combines machine learning with explainable
            AI to provide an easy-to-understand thyroid disease
            prediction and insights into the factors influencing
            the result.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => navigate("predict")}
            >
              Start Prediction
              <ArrowRight size={21} />
            </button>

            <button
              className="secondary-button"
              onClick={() => navigate("about")}
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
              <small>Prediction model</small>
            </span>
          </div>

          <div className="floating-card floating-two">
            <Sparkles size={25} />

            <span>
              <strong>SHAP</strong>
              <small>Explainable result</small>
            </span>
          </div>

          <div className="medical-card">
            <div className="card-top">
              <span className="status-dot" />
              AI DIAGNOSTIC ENGINE

              <Activity size={23} />
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
                <small>MODEL STATUS</small>
                <strong>Ready for analysis</strong>
              </div>

              <CheckCircle2 size={25} />
            </div>
          </div>
        </div>
      </section>
    </>
  );

  /* =======================================================
     PREDICTION FORM
  ======================================================= */

  const PredictionPage = () => (
    <main className="page-section">
      <button
        className="back-button"
        onClick={() => navigate("home")}
      >
        <ArrowLeft size={17} />
        Back to home
      </button>

      <div className="page-heading">
        <div className="eyebrow">
          <Stethoscope size={15} />
          THYROID PREDICTION
        </div>

        <h2>Enter clinical information</h2>

        <p>
          Provide the available patient information below. The
          trained machine learning model will analyze the values
          and generate an explainable prediction.
        </p>
      </div>

      <form
        className="prediction-form"
        onSubmit={handlePrediction}
      >
        <div className="form-card">
          <div className="form-card-heading">
            <div className="section-icon">
              <Activity size={22} />
            </div>

            <div>
              <h3>Clinical measurements</h3>
              <p>
                Enter the patient's basic measurements and thyroid
                laboratory values.
              </p>
            </div>
          </div>

          <div className="form-grid">
            {numericFields.map(
              ([key, label, min, max, step]) => (
                <label className="input-group" key={key}>
                  <span>{label}</span>

                  <input
                    type="number"
                    value={form[key]}
                    min={min}
                    max={max}
                    step={step}
                    onChange={(event) =>
                      updateField(key, event.target.value)
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
              <ShieldCheck size={22} />
            </div>

            <div>
              <h3>Clinical indicators</h3>
              <p>
                Select 0 for No and 1 for Yes for each indicator.
              </p>
            </div>
          </div>

          <div className="indicator-grid">
            {binaryFields.map(([key, label]) => (
              <label className="toggle-row" key={key}>
                <span>{label}</span>

                <select
                  value={form[key]}
                  onChange={(event) =>
                    updateField(key, Number(event.target.value))
                  }
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </label>
            ))}
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
              <Activity className="spin" size={20} />
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
            This tool is intended for educational and research
            purposes. Its prediction should not be considered a
            medical diagnosis or a replacement for a qualified
            healthcare professional.
          </span>
        </div>
      </form>
    </main>
  );

  /* =======================================================
     RESULTS
  ======================================================= */

  const ResultsPage = () => {
    if (!result) {
      return (
        <main className="page-section">
          <div className="empty-result">
            <div className="page-heading">
              <div className="eyebrow">
                <Info size={15} />
                NO RESULT
              </div>

              <h2>No prediction available</h2>

              <p>
                Please enter the clinical information and run a
                prediction first.
              </p>

              <button
                className="primary-button"
                onClick={() => navigate("predict")}
              >
                Start Prediction
                <ArrowRight size={19} />
              </button>
            </div>
          </div>
        </main>
      );
    }

    const positive = result.prediction === 1;

    return (
      <main className="page-section">
        <button
          className="back-button"
          onClick={() => navigate("predict")}
        >
          <ArrowLeft size={17} />
          New prediction
        </button>

        <div className="page-heading">
          <div className="eyebrow">
            <Sparkles size={15} />
            YOUR PREDICTION RESULT
          </div>

          <h2>AI analysis completed</h2>

          <p>
            The machine learning model has analyzed the submitted
            clinical information.
          </p>
        </div>

        <div
          className={`result-banner ${
            positive ? "positive" : "negative"
          }`}
        >
          <div className="result-icon">
            {positive ? (
              <CircleAlert size={34} />
            ) : (
              <CheckCircle2 size={34} />
            )}
          </div>

          <div>
            <span>MODEL PREDICTION</span>

            <h2>{result.prediction_label}</h2>

            <p>
              Model: <strong>{result.model}</strong>
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
                <span>Class 0</span>
                <strong>
                  {formatPercentage(
                    result.probabilities.class_0
                  )}
                </strong>
              </div>

              <div className="progress">
                <div
                  className="progress-bar"
                  style={{
                    width: `${percentage(
                      result.probabilities.class_0
                    )}%`,
                  }}
                />
              </div>
            </div>

            <div className="probability">
              <div className="probability-head">
                <span>Class 1</span>
                <strong>
                  {formatPercentage(
                    result.probabilities.class_1
                  )}
                </strong>
              </div>

              <div className="progress">
                <div
                  className="progress-bar"
                  style={{
                    width: `${percentage(
                      result.probabilities.class_1
                    )}%`,
                  }}
                />
              </div>
            </div>

            <p className="result-text">
              Class 0 represents no thyroid disease prediction.
              Class 1 represents a thyroid disease prediction.
            </p>
          </section>

          <section className="result-card">
            <div className="result-card-title">
              <Sparkles size={19} />
              Explainable AI
            </div>

            <p className="result-text">
              SHAP explains which input features contributed to
              the model's prediction. Larger absolute values
              generally indicate stronger influence on the model
              output.
            </p>

            <div className="shap-list">
              {result.shap.length > 0 ? (
                result.shap.map((item, index) => (
                  <div
                    className="shap-row"
                    key={`${item.feature}-${index}`}
                  >
                    <span>{String(item.feature)}</span>

                    <strong>
                      {safeNumber(item.value).toFixed(4)}
                    </strong>
                  </div>
                ))
              ) : (
                <div className="empty-small">
                  SHAP explanation will appear here when returned
                  by the backend.
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
            Counterfactual explanations describe changes to the
            input values that could potentially change the model's
            prediction.
          </p>

          <div className="counterfactual-list">
            {result.counterfactuals.length > 0 ? (
              result.counterfactuals.map((item, index) => (
                <div
                  className="counterfactual-item"
                  key={index}
                >
                  {String(item)}
                </div>
              ))
            ) : (
              <div className="empty-small">
                Counterfactual explanation will appear here when
                returned by the backend.
              </div>
            )}
          </div>
        </section>

        <div className="result-actions">
          <button
            className="primary-button"
            onClick={downloadReport}
          >
            <Download size={18} />
            Download Report
          </button>
        </div>

        <div className="medical-disclaimer">
          <Info size={18} />

          <span>
            This prediction is generated by a machine learning
            model for educational and research purposes. Please
            consult a qualified healthcare professional for
            medical interpretation.
          </span>
        </div>
      </main>
    );
  };

  /* =======================================================
     ABOUT
  ======================================================= */

  const AboutPage = () => (
    <main className="page-section">
      <div className="page-heading">
        <div className="eyebrow">
          <Sparkles size={15} />
          ABOUT THYROCARE
        </div>

        <h2>AI with understandable results.</h2>

        <p>
          ThyroCare combines machine learning prediction with
          explainable artificial intelligence to make model
          outputs easier to understand.
        </p>
      </div>

      <div className="about-grid">
        <section className="about-card">
          <Brain size={30} />

          <h3>Machine Learning</h3>

          <p>
            The application uses a trained XGBoost machine
            learning model to analyze the submitted thyroid
            clinical features.
          </p>
        </section>

        <section className="about-card">
          <Sparkles size={30} />

          <h3>Explainable AI</h3>

          <p>
            SHAP-based explanations can show which features had
            stronger influence on the model output.
          </p>
        </section>

        <section className="about-card">
          <ShieldCheck size={30} />

          <h3>Transparent Results</h3>

          <p>
            The results page displays the predicted class and
            class probabilities instead of showing only a single
            prediction.
          </p>
        </section>

        <section className="about-card">
          <HeartPulse size={30} />

          <h3>Research Project</h3>

          <p>
            ThyroCare is designed as an academic machine learning
            and explainable AI project for thyroid disease
            analysis.
          </p>
        </section>
      </div>

      <div className="about-cta">
        <div>
          <h3>Ready to run an analysis?</h3>

          <p>
            Enter the clinical information and explore the model
            prediction.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={() => navigate("predict")}
        >
          Start Prediction
          <ArrowRight size={19} />
        </button>
      </div>
    </main>
  );

  /* =======================================================
     FOOTER
  ======================================================= */

  const Footer = () => (
    <footer className="footer">
      <div className="footer-brand">
        <span className="brand-icon">
          <HeartPulse size={20} />
        </span>

        <div>
          <strong>ThyroCare AI</strong>

          <p>
            Intelligent thyroid disease prediction.
          </p>
        </div>
      </div>

      <div className="footer-note">
        <ShieldCheck size={15} />
        For educational and research purposes.
      </div>
    </footer>
  );

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="app-shell">
      <Navbar />

      {page === "home" && <HomePage />}

      {page === "predict" && <PredictionPage />}

      {page === "results" && <ResultsPage />}

      {page === "about" && <AboutPage />}

      <Footer />
    </div>
  );
         }
