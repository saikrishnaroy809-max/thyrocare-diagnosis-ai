import React, { useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronRight,
  Download,
  HeartPulse,
  Info,
  Loader2,
  Home as HomeIcon,
  Menu,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  X,
  Zap,
  AlertCircle,
} from "lucide-react";

/* =========================================================
   BACKEND
========================================================= */

const API_URL =
  "https://thyrocare-diagnosis-ai.onrender.com";

/* =========================================================
   DEFAULT FORM
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
  T4U: 1,
  "FTI measured": 1,
  FTI: 110,
};

const numericFields = [
  "age",
  "TSH",
  "TT4",
  "T4U",
  "FTI",
];

/* =========================================================
   FIELD GROUPS
========================================================= */

const featureGroups = [
  {
    title: "Patient Information",
    icon: <HeartPulse size={18} />,
    fields: ["age", "sex"],
  },
  {
    title: "Medical History",
    icon: <Stethoscope size={18} />,
    fields: [
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
    ],
  },
  {
    title: "Thyroid Measurements",
    icon: <Activity size={18} />,
    fields: [
      "TSH measured",
      "TSH",
      "T3 measured",
      "TT4 measured",
      "TT4",
      "T4U measured",
      "T4U",
      "FTI measured",
      "FTI",
    ],
  },
];

/* =========================================================
   DISPLAY LABELS
========================================================= */

const labels = {
  age: "Age",
  sex: "Sex",
  "on thyroxine": "On Thyroxine",
  "query on thyroxine": "Query on Thyroxine",
  "on antithyroid medication":
    "On Antithyroid Medication",
  sick: "Sick",
  pregnant: "Pregnant",
  "thyroid surgery": "Thyroid Surgery",
  "I131 treatment": "I131 Treatment",
  "query hypothyroid": "Query Hypothyroid",
  "query hyperthyroid": "Query Hyperthyroid",
  lithium: "Lithium",
  goitre: "Goitre",
  tumor: "Tumor",
  hypopituitary: "Hypopituitary",
  psych: "Psych",
  "TSH measured": "TSH Measured",
  TSH: "TSH",
  "T3 measured": "T3 Measured",
  "TT4 measured": "TT4 Measured",
  TT4: "TT4",
  "T4U measured": "T4U Measured",
  T4U: "T4U",
  "FTI measured": "FTI Measured",
  FTI: "FTI",
};

/* =========================================================
   HELPERS
========================================================= */

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function percent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return 0;
  }

  return number <= 1
    ? number * 100
    : number;
}

function percentText(value) {
  return `${percent(value).toFixed(2)}%`;
}

/* =========================================================
   ERROR FORMATTER
   Prevents [object Object]
========================================================= */

function formatApiError(data, status) {
  if (!data) {
    return `Server returned HTTP ${status}.`;
  }

  if (typeof data === "string") {
    return data;
  }

  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        const location = Array.isArray(item?.loc)
          ? item.loc.join(" → ")
          : "";

        const message =
          item?.msg ||
          item?.message ||
          item?.detail ||
          JSON.stringify(item);

        return location
          ? `${location}: ${message}`
          : message;
      })
      .join("\n");
  }

  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (data.detail) {
    return JSON.stringify(
      data.detail,
      null,
      2
    );
  }

  if (data.message) {
    return String(data.message);
  }

  return `Server returned HTTP ${status}.`;
}

/* =========================================================
   PREDICTION HELPERS
========================================================= */

function getPrediction(data) {
  const value =
    data?.prediction ??
    data?.predicted_class ??
    data?.class ??
    data?.result ??
    0;

  if (typeof value === "string") {
    const lower = value.toLowerCase();

    if (
      lower.includes("predicted") &&
      !lower.includes("not predicted")
    ) {
      return 1;
    }

    if (
      lower.includes("disease") &&
      !lower.includes("not")
    ) {
      return 1;
    }
  }

  return Number(value) === 1 ? 1 : 0;
}

function getModel(data) {
  return (
    data?.model ??
    data?.model_name ??
    data?.algorithm ??
    "XGBoost"
  );
}

function getProbability(data, classNumber) {
  const probabilities =
    data?.probabilities ??
    data?.class_probabilities ??
    data?.prediction_probabilities ??
    {};

  return (
    probabilities[`class_${classNumber}`] ??
    probabilities[classNumber] ??
    data?.[`class_${classNumber}_probability`] ??
    0
  );
}

/* =========================================================
   SHAP
========================================================= */

function normalizeShap(data) {
  const raw =
    data?.shap ??
    data?.shap_values ??
    data?.explanation ??
    data?.feature_importance ??
    [];

  if (Array.isArray(raw)) {
    return raw
      .map((item) => {
        if (typeof item === "number") {
          return {
            feature: "Feature",
            impact: item,
          };
        }

        return {
          feature:
            item?.feature ??
            item?.name ??
            item?.feature_name ??
            "Feature",

          impact:
            Number(
              item?.impact ??
                item?.shap_value ??
                item?.value ??
                0
            ) || 0,
        };
      })
      .sort(
        (a, b) =>
          Math.abs(b.impact) -
          Math.abs(a.impact)
      );
  }

  if (
    typeof raw === "object" &&
    raw !== null
  ) {
    return Object.entries(raw)
      .map(([feature, value]) => ({
        feature,
        impact: Number(value) || 0,
      }))
      .sort(
        (a, b) =>
          Math.abs(b.impact) -
          Math.abs(a.impact)
      );
  }

  return [];
}

/* =========================================================
   COUNTERFACTUALS
========================================================= */

function normalizeCounterfactuals(data) {
  const raw =
    data?.counterfactuals ??
    data?.counterfactual ??
    data?.counterfactual_explanations ??
    data?.cf_explanation ??
    null;

  if (!raw) {
    return {
      available: false,
      features: [],
      scenarios: [],
      target_prediction: null,
      target_label: "",
    };
  }

  if (
    typeof raw === "object" &&
    !Array.isArray(raw)
  ) {
    return {
      available: Boolean(raw.available),

      features: Array.isArray(
        raw.features
      )
        ? raw.features
        : [],

      scenarios: Array.isArray(
        raw.scenarios
      )
        ? raw.scenarios
        : [],

      target_prediction:
        raw.target_prediction ??
        null,

      target_label:
        raw.target_label ?? "",
    };
  }

  if (Array.isArray(raw)) {
    return {
      available: raw.length > 0,
      features: [],
      scenarios: raw,
      target_prediction: null,
      target_label: "",
    };
  }

  return {
    available: false,
    features: [],
    scenarios: [],
    target_prediction: null,
    target_label: "",
  };
}

/* =========================================================
   PAYLOAD
========================================================= */

function buildPayload(form) {
  return {
    age: toNumber(form.age),
    sex: toNumber(form.sex),

    "on thyroxine": toNumber(
      form["on thyroxine"]
    ),

    "query on thyroxine": toNumber(
      form["query on thyroxine"]
    ),

    "on antithyroid medication":
      toNumber(
        form["on antithyroid medication"]
      ),

    sick: toNumber(form.sick),
    pregnant: toNumber(form.pregnant),

    "thyroid surgery": toNumber(
      form["thyroid surgery"]
    ),

    I131_treatment: toNumber(
      form["I131 treatment"]
    ),

    "query hypothyroid": toNumber(
      form["query hypothyroid"]
    ),

    "query hyperthyroid": toNumber(
      form["query hyperthyroid"]
    ),

    lithium: toNumber(form.lithium),
    goitre: toNumber(form.goitre),
    tumor: toNumber(form.tumor),

    hypopituitary: toNumber(
      form.hypopituitary
    ),

    psych: toNumber(form.psych),

    TSH_measured: toNumber(
      form["TSH measured"]
    ),

    TSH: toNumber(form.TSH),

    T3_measured: toNumber(
      form["T3 measured"]
    ),

    TT4_measured: toNumber(
      form["TT4 measured"]
    ),

    TT4: toNumber(form.TT4),

    T4U_measured: toNumber(
      form["T4U measured"]
    ),

    T4U: toNumber(form.T4U),

    FTI_measured: toNumber(
      form["FTI measured"]
    ),

    FTI: toNumber(form.FTI),
  };
}

/* =========================================================
   APP
========================================================= */

export default function App() {
  const [page, setPage] =
    useState("home");

  const [mobileMenu, setMobileMenu] =
    useState(false);

  const [form, setForm] =
    useState(defaultForm);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  /* =====================================================
     NAVIGATION
  ===================================================== */

  const navigate = (target) => {
    setPage(target);
    setMobileMenu(false);
  };

  /* =====================================================
     FORM CHANGE
  ===================================================== */

  const handleChange = (
    field,
    value
  ) => {
    setForm((previous) => ({
      ...previous,
      [field]: numericFields.includes(
        field
      )
        ? value
        : Number(value),
    }));
  };

  /* =====================================================
     RESET
  ===================================================== */

  const resetForm = () => {
    setForm(defaultForm);
    setResult(null);
    setError("");
  };

  /* =====================================================
     PREDICTION
  ===================================================== */

  const handlePrediction =
    async () => {
      setLoading(true);
      setError("");
      setResult(null);

      const controller =
        new AbortController();

      const timeoutId =
        setTimeout(() => {
          controller.abort();
        }, 180000);

      try {
        const payload =
          buildPayload(form);

        console.log(
          "Sending prediction payload:",
          payload
        );

        const response =
          await fetch(
            `${API_URL}/predict`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",

                Accept:
                  "application/json",
              },

              body: JSON.stringify(
                payload
              ),

              signal:
                controller.signal,
            }
          );

        clearTimeout(timeoutId);

        let data = null;

        try {
          data =
            await response.json();
        } catch {
          throw new Error(
            "The prediction server returned an invalid response."
          );
        }

        console.log(
          "Prediction response:",
          data
        );

        if (!response.ok) {
          throw new Error(
            formatApiError(
              data,
              response.status
            )
          );
        }

        const prediction =
          getPrediction(data);

        const normalized = {
          prediction,

          prediction_label:
            data?.prediction_label ??
            (prediction === 1
              ? "Thyroid Disease Predicted"
              : "Thyroid Disease Not Predicted"),

          model:
            getModel(data),

          probabilities: {
            class_0:
              getProbability(
                data,
                0
              ),

            class_1:
              getProbability(
                data,
                1
              ),
          },

          shap:
            normalizeShap(data),

          counterfactuals:
            normalizeCounterfactuals(
              data
            ),

          raw: data,
        };

        setResult(normalized);
        setPage("results");
      } catch (err) {
        clearTimeout(timeoutId);

        console.error(
          "Prediction error:",
          err
        );

        if (
          err.name ===
          "AbortError"
        ) {
          setError(
            "The prediction server took too long to respond. Render may be waking up. Please try again."
          );
        } else {
          setError(
            err?.message ||
              "Prediction failed. Please try again."
          );
        }
      } finally {
        setLoading(false);
      }
    };

  /* =====================================================
     REPORT
  ===================================================== */

  const downloadReport =
    () => {
      if (!result) return;

      const cf =
        result.counterfactuals;

      let report =
        "THYROCARE AI - THYROID ANALYSIS REPORT\n";

      report +=
        "========================================\n\n";

      report += `Model: ${result.model}\n`;

      report += `Prediction: ${result.prediction_label}\n`;

      report += `Class 0 Probability: ${percentText(
        result.probabilities
          .class_0
      )}\n`;

      report += `Class 1 Probability: ${percentText(
        result.probabilities
          .class_1
      )}\n\n`;

      report +=
        "SHAP FEATURE CONTRIBUTIONS\n";
      report +=
        "---------------------------\n";

      result.shap.forEach(
        (item) => {
          report += `${item.feature}: ${Number(
            item.impact
          ).toFixed(6)}\n`;
        }
      );

      report +=
        "\nCOUNTERFACTUAL SCENARIOS\n";
      report +=
        "------------------------\n";

      if (
        cf?.available &&
        cf.scenarios?.length
      ) {
        cf.scenarios.forEach(
          (scenario) => {
            report += `Scenario ${scenario.scenario}\n`;

            if (
              Array.isArray(
                scenario.changes
              )
            ) {
              scenario.changes.forEach(
                (change) => {
                  report += `  ${change.feature}: ${change.original_value} -> ${change.counterfactual_value}\n`;
                }
              );
            }

            report += `  Prediction: ${
              scenario.prediction_label ??
              scenario.prediction
            }\n`;

            report += `  Class 0: ${percentText(
              scenario.class_0_probability
            )}\n`;

            report += `  Class 1: ${percentText(
              scenario.class_1_probability
            )}\n\n`;
          }
        );
      } else {
        report +=
          "No counterfactual scenarios available.\n";
      }

      report +=
        "\nEducational and research purposes only. This output should not replace professional medical advice.\n";

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

      document.body.appendChild(
        link
      );

      link.click();

      link.remove();

      URL.revokeObjectURL(url);
    };

  return (
    <div className="app">
      {/* NAVBAR */}

      <header className="navbar">
        <div
          className="brand"
          onClick={() =>
            navigate("home")
          }
        >
          <div className="brand-icon">
            <Activity size={24} />
          </div>

          <div>
            <div className="brand-name">
              ThyroCare AI
            </div>

            <div className="brand-subtitle">
              Intelligent thyroid analysis
            </div>
          </div>
        </div>

        <nav className="desktop-nav">
          <button
            className={
              page === "home"
                ? "nav-link active"
                : "nav-link"
            }
            onClick={() =>
              navigate("home")
            }
          >
            <HomeIcon size={16} />
            Home
          </button>

          <button
            className={
              page === "prediction"
                ? "nav-link active"
                : "nav-link"
            }
            onClick={() =>
              navigate(
                "prediction"
              )
            }
          >
            <Brain size={16} />
            Prediction
          </button>

          {result && (
            <button
              className={
                page === "results"
                  ? "nav-link active"
                  : "nav-link"
              }
              onClick={() =>
                navigate("results")
              }
            >
              <Activity size={16} />
              Results
            </button>
          )}

          <button
            className={
              page === "about"
                ? "nav-link active"
                : "nav-link"
            }
            onClick={() =>
              navigate("about")
            }
          >
            <Info size={16} />
            About
          </button>
        </nav>

        <button
          className="mobile-menu-button"
          onClick={() =>
            setMobileMenu(
              !mobileMenu
            )
          }
        >
          {mobileMenu ? (
            <X />
          ) : (
            <Menu />
          )}
        </button>
      </header>

      {mobileMenu && (
        <div className="mobile-nav">
          <button
            onClick={() =>
              navigate("home")
            }
          >
            Home
          </button>

          <button
            onClick={() =>
              navigate(
                "prediction"
              )
            }
          >
            Prediction
          </button>

          {result && (
            <button
              onClick={() =>
                navigate("results")
              }
            >
              Results
            </button>
          )}

          <button
            onClick={() =>
              navigate("about")
            }
          >
            About
          </button>
        </div>
      )}

      {/* HOME */}

      {page === "home" && (
        <main>
          <section className="hero">
            <div className="hero-content">
              <div className="hero-badge">
                <Sparkles size={16} />
                AI-Powered Thyroid Analysis
              </div>

              <h1>
                Intelligent
                <span>
                  {" "}
                  Thyroid Analysis
                </span>
              </h1>

              <p>
                Analyze thyroid-related
                clinical features using
                machine learning and
                explainable AI.
              </p>

              <div className="hero-actions">
                <button
                  className="primary-button"
                  onClick={() =>
                    navigate(
                      "prediction"
                    )
                  }
                >
                  Start Analysis
                  <ChevronRight
                    size={18}
                  />
                </button>

                <button
                  className="secondary-button"
                  onClick={() =>
                    navigate(
                      "about"
                    )
                  }
                >
                  Learn More
                </button>
              </div>

              <div className="trust-row">
                <div>
                  <ShieldCheck
                    size={18}
                  />
                  Explainable AI
                </div>

                <div>
                  <Zap size={18} />
                  XGBoost
                </div>

                <div>
                  <Brain size={18} />
                  SHAP
                </div>
              </div>
            </div>

            <div className="hero-visual">
              <div className="orb">
                <Activity size={80} />
              </div>
            </div>
          </section>

          <section className="features-section">
            <div className="section-heading">
              <span>
                POWERED BY AI
              </span>

              <h2>
                From prediction to
                explanation
              </h2>

              <p>
                Machine learning,
                explainable AI and
                counterfactual analysis
                in one application.
              </p>
            </div>

            <div className="feature-grid">
              <FeatureCard
                icon={<Brain />}
                title="Machine Learning"
                text="Uses an XGBoost classification model for thyroid-related prediction."
              />

              <FeatureCard
                icon={<Sparkles />}
                title="SHAP Explainability"
                text="Shows which input features contributed to the model output."
              />

              <FeatureCard
                icon={<Zap />}
                title="Counterfactual Analysis"
                text="Explores hypothetical input changes and their effect on the model."
              />
            </div>
          </section>
        </main>
      )}

      {/* PREDICTION */}

      {page === "prediction" && (
        <PredictionPage
          form={form}
          handleChange={handleChange}
          handlePrediction={
            handlePrediction
          }
          resetForm={resetForm}
          loading={loading}
          error={error}
        />
      )}

      {/* RESULTS */}

      {page === "results" && (
        <ResultsPage
          result={result}
          downloadReport={
            downloadReport
          }
          onNewPrediction={() => {
            resetForm();
            setPage("prediction");
          }}
        />
      )}

      {/* ABOUT */}

      {page === "about" && (
        <main className="page-container">
          <section className="about-card">
            <div className="section-icon">
              <Brain size={32} />
            </div>

            <h1>
              About ThyroCare AI
            </h1>

            <p>
              ThyroCare AI is an
              educational and research
              project demonstrating
              machine learning and
              explainable AI for thyroid
              disease classification.
            </p>

            <div className="about-grid">
              <InfoCard
                title="Machine Learning"
                text="The application uses an XGBoost classification model."
              />

              <InfoCard
                title="Explainable AI"
                text="SHAP values provide feature-level explanations."
              />

              <InfoCard
                title="Counterfactuals"
                text="The application explores hypothetical feature changes."
              />

              <InfoCard
                title="Research Purpose"
                text="Designed for academic, educational and research demonstration."
              />
            </div>

            <div className="disclaimer">
              <AlertCircle size={20} />

              <div>
                <strong>
                  Important
                </strong>

                <p>
                  This application is for
                  educational and research
                  purposes only and should
                  not replace advice from a
                  qualified healthcare
                  professional.
                </p>
              </div>
            </div>
          </section>
        </main>
      )}

      {/* FOOTER */}

      <footer className="footer">
        <div>
          <strong>
            ThyroCare AI
          </strong>

          <span>
            Intelligent thyroid analysis
          </span>
        </div>

        <p>
          Educational and research
          purposes only.
        </p>
      </footer>
    </div>
  );
}

/* =========================================================
   FEATURE CARD
========================================================= */

function FeatureCard({
  icon,
  title,
  text,
}) {
  return (
    <div className="feature-card">
      <div className="feature-icon">
        {icon}
      </div>

      <h3>{title}</h3>

      <p>{text}</p>
    </div>
  );
}

/* =========================================================
   INFO CARD
========================================================= */

function InfoCard({
  title,
  text,
}) {
  return (
    <div className="info-card">
      <h3>{title}</h3>

      <p>{text}</p>
    </div>
  );
}

/* =========================================================
   PREDICTION PAGE
========================================================= */

function PredictionPage({
  form,
  handleChange,
  handlePrediction,
  resetForm,
  loading,
  error,
}) {
  return (
    <main className="page-container">
      <div className="page-heading">
        <div className="section-icon">
          <Brain size={28} />
        </div>

        <div>
          <h1>
            Thyroid Prediction
          </h1>

          <p>
            Enter the clinical features
            and run the machine learning
            analysis.
          </p>
        </div>
      </div>

      {error && (
        <div className="error-box">
          <AlertCircle size={22} />

          <div>
            <strong>
              Prediction Error
            </strong>

            <p
              style={{
                whiteSpace:
                  "pre-line",
              }}
            >
              {error}
            </p>

            <small>
              If this is the first request
              after inactivity, Render may
              need some time to wake up.
            </small>
          </div>
        </div>
      )}

      <div className="prediction-layout">
        <div className="form-card">
          {featureGroups.map(
            (group) => (
              <div
                className="form-section"
                key={group.title}
              >
                <div className="form-section-title">
                  {group.icon}

                  <h2>
                    {group.title}
                  </h2>
                </div>

                <div className="input-grid">
                  {group.fields.map(
                    (field) => {
                      const isNumeric =
                        numericFields.includes(
                          field
                        );

                      return (
                        <div
                          className="input-group"
                          key={field}
                        >
                          <label>
                            {labels[field] ||
                              field}
                          </label>

                          {isNumeric ? (
                            <input
                              type="number"
                              step={
                                field ===
                                  "TSH" ||
                                field ===
                                  "T4U"
                                  ? "0.01"
                                  : "any"
                              }
                              value={
                                form[field]
                              }
                              onChange={(
                                e
                              ) =>
                                handleChange(
                                  field,
                                  e.target
                                    .value
                                )
                              }
                            />
                          ) : (
                            <select
                              value={
                                form[field]
                              }
                              onChange={(
                                e
                              ) =>
                                handleChange(
                                  field,
                                  e.target
                                    .value
                                )
                              }
                            >
                              <option value={0}>
                                No / 0
                              </option>

                              <option value={1}>
                                Yes / 1
                              </option>
                            </select>
                          )}
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            )
          )}

          <div className="form-actions">
            <button
              className="secondary-button"
              onClick={resetForm}
              disabled={loading}
            >
              <RotateCcw size={17} />
              Reset
            </button>

            <button
              className="primary-button"
              onClick={
                handlePrediction
              }
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2
                    size={18}
                    className="spin"
                  />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Analyze Thyroid
                </>
              )}
            </button>
          </div>

          {loading && (
            <div className="loading-message">
              <Loader2
                size={18}
                className="spin"
              />

              <span>
                Running prediction, SHAP
                and counterfactual analysis.
                The first request may take
                longer while Render wakes up.
              </span>
            </div>
          )}
        </div>

        <aside className="side-info">
          <div className="side-card">
            <ShieldCheck size={25} />

            <h3>
              Explainable Analysis
            </h3>

            <p>
              SHAP feature explanations
              and hypothetical
              counterfactual scenarios are
              provided with the prediction.
            </p>
          </div>

          <div className="side-card">
            <Info size={25} />

            <h3>
              Educational Use
            </h3>

            <p>
              Results are intended for
              educational and research
              purposes only.
            </p>
          </div>
        </aside>
      </div>
    </main>
  );
}

/* =========================================================
   RESULTS PAGE
========================================================= */

function ResultsPage({
  result,
  downloadReport,
  onNewPrediction,
}) {
  if (!result) {
    return (
      <main className="page-container">
        <div className="empty-results">
          <Brain size={50} />

          <h2>
            No prediction yet
          </h2>

          <button
            className="primary-button"
            onClick={
              onNewPrediction
            }
          >
            Start Prediction
          </button>
        </div>
      </main>
    );
  }

  const disease =
    result.prediction === 1;

  const cf =
    result.counterfactuals;

  return (
    <main className="page-container">
      <div className="results-header">
        <div>
          <div className="section-icon">
            <Activity size={28} />
          </div>

          <h1>
            Analysis Results
          </h1>

          <p>
            Model prediction and
            explainable AI analysis.
          </p>
        </div>

        <div className="result-actions">
          <button
            className="secondary-button"
            onClick={
              onNewPrediction
            }
          >
            <RotateCcw size={17} />
            New Prediction
          </button>

          <button
            className="primary-button"
            onClick={
              downloadReport
            }
          >
            <Download size={17} />
            Download Report
          </button>
        </div>
      </div>

      {/* PREDICTION */}

      <section
        className={
          disease
            ? "result-banner positive"
            : "result-banner negative"
        }
      >
        <div className="result-symbol">
          {disease ? (
            <AlertCircle size={38} />
          ) : (
            <CheckCircle2 size={38} />
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
      </section>

      {/* PROBABILITIES */}

      <section className="results-grid">
        <ProbabilityCard
          title="Class 0"
          value={
            result.probabilities
              .class_0
          }
          description="Not Predicted"
        />

        <ProbabilityCard
          title="Class 1"
          value={
            result.probabilities
              .class_1
          }
          description="Thyroid Disease Predicted"
        />
      </section>

      {/* SHAP */}

      <section className="results-section">
        <div className="results-section-heading">
          <Brain size={22} />

          <div>
            <h2>
              SHAP Explanation
            </h2>

            <p>
              Features with the largest
              absolute contribution to the
              model output.
            </p>
          </div>
        </div>

        {result.shap.length === 0 ? (
          <div className="empty-box">
            No SHAP values were returned.
          </div>
        ) : (
          <div className="shap-list">
            {result.shap
              .slice(0, 10)
              .map(
                (item, index) => {
                  const impact =
                    Number(
                      item.impact
                    ) || 0;

                  return (
                    <div
                      className="shap-item"
                      key={`${item.feature}-${index}`}
                    >
                      <div className="shap-info">
                        <span>
                          {item.feature}
                        </span>

                        <strong>
                          {impact > 0
                            ? "+"
                            : ""}
                          {impact.toFixed(
                            4
                          )}
                        </strong>
                      </div>

                      <div className="shap-track">
                        <div
                          className="shap-value"
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(
                                3,
                                Math.abs(
                                  impact
                                ) * 20
                              )
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                }
              )}
          </div>
        )}
      </section>

      {/* COUNTERFACTUALS */}

      <section className="results-section">
        <div className="results-section-heading">
          <Sparkles size={22} />

          <div>
            <h2>
              Counterfactual Explanation
            </h2>

            <p>
              Hypothetical input changes
              explored by the model.
            </p>
          </div>
        </div>

        {!cf?.available ? (
          <div className="empty-box">
            Counterfactual analysis is not
            available for this prediction.
          </div>
        ) : (
          <>
            {cf.target_label && (
              <div className="target-box">
                <span>
                  Target model output
                </span>

                <strong>
                  {cf.target_label}
                </strong>
              </div>
            )}

            {cf.features?.length >
              0 && (
              <div className="cf-feature-box">
                <h3>
                  Key Change
                </h3>

                {cf.features.map(
                  (
                    feature,
                    index
                  ) => (
                    <div
                      className="cf-change"
                      key={index}
                    >
                      <span>
                        {
                          feature.feature
                        }
                      </span>

                      <strong>
                        {
                          feature.original_value
                        }{" "}
                        →
                        {" "}
                        {
                          feature.counterfactual_value
                        }
                      </strong>
                    </div>
                  )
                )}
              </div>
            )}

            {cf.scenarios?.length >
              0 && (
              <div className="counterfactual-list">
                {cf.scenarios.map(
                  (scenario) => (
                    <div
                      className="counterfactual-item"
                      key={
                        scenario.scenario
                      }
                    >
                      <div className="scenario-header">
                        <div>
                          <span>
                            SCENARIO{" "}
                            {
                              scenario.scenario
                            }
                          </span>

                          <h3>
                            {scenario.prediction_label ??
                              `Class ${scenario.prediction}`}
                          </h3>
                        </div>

                        <Sparkles size={20} />
                      </div>

                      {Array.isArray(
                        scenario.changes
                      ) &&
                        scenario.changes
                          .length >
                          0 && (
                          <div className="scenario-changes">
                            {scenario.changes.map(
                              (
                                change,
                                index
                              ) => (
                                <div
                                  className="scenario-change"
                                  key={
                                    index
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
                                    →
                                    {" "}
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
                            {percentText(
                              scenario.class_0_probability
                            )}
                          </strong>
                        </div>

                        <div>
                          <span>
                            Class 1
                          </span>

                          <strong>
                            {percentText(
                              scenario.class_1_probability
                            )}
                          </strong>
                        </div>

                        {scenario.distance !==
                          undefined && (
                          <div>
                            <span>
                              Distance
                            </span>

                            <strong>
                              {
                                scenario.distance
                              }
                            </strong>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                )}
              </div>
            )}

            <div className="cf-disclaimer">
              <Info size={17} />

              <span>
                These are hypothetical
                machine-learning sensitivity
                scenarios. They are not medical
                recommendations.
              </span>
            </div>
          </>
        )}
      </section>

      <div className="disclaimer">
        <ShieldCheck size={20} />

        <div>
          <strong>
            Educational and research
            purposes only
          </strong>

          <p>
            This model output should not
            replace professional medical
            evaluation, diagnosis or
            treatment.
          </p>
        </div>
      </div>
    </main>
  );
}

/* =========================================================
   PROBABILITY CARD
========================================================= */

function ProbabilityCard({
  title,
  value,
  description,
}) {
  const width = Math.min(
    100,
    Math.max(0, percent(value))
  );

  return (
    <div className="result-card">
      <div className="result-card-top">
        <span>{title}</span>

        <strong>
          {percentText(value)}
        </strong>
      </div>

      <div className="progress">
        <div
          className="progress-bar"
          style={{
            width: `${width}%`,
          }}
        />
      </div>

      <p>{description}</p>
    </div>
  );
          }
