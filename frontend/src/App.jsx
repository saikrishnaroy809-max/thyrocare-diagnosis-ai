import React, { useState } from "react";
import {
  Activity,
  Brain,
  ChevronDown,
  ChevronUp,
  Download,
  HeartPulse,
  Info,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  TrendingUp,
} from "lucide-react";
import "./index.css";

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

const labels = {
  age: "Age",
  sex: "Sex",
  "on thyroxine": "On Thyroxine",
  "query on thyroxine": "Query on Thyroxine",
  "on antithyroid medication": "On Antithyroid Medication",
  sick: "Sick",
  pregnant: "Pregnant",
  "thyroid surgery": "Thyroid Surgery",
  "I131 treatment": "I-131 Treatment",
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

const numericFields = ["age", "TSH", "TT4", "T4U", "FTI"];

const groups = [
  {
    title: "Patient Information",
    icon: <HeartPulse size={20} />,
    fields: ["age", "sex"],
  },
  {
    title: "Medical History",
    icon: <Stethoscope size={20} />,
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
    icon: <Activity size={20} />,
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

function numberValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/* Convert FastAPI validation errors into readable text */
function formatApiError(data, status) {
  if (!data) {
    return `Request failed with status ${status}.`;
  }

  if (Array.isArray(data.detail)) {
    return data.detail
      .map((error, index) => {
        const location = Array.isArray(error?.loc)
          ? error.loc.join(" → ")
          : "field";

        const message =
          error?.msg ||
          error?.message ||
          "Invalid value";

        return `${index + 1}. ${location}: ${message}`;
      })
      .join("\n");
  }

  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (data.detail && typeof data.detail === "object") {
    return JSON.stringify(data.detail, null, 2);
  }

  if (typeof data.message === "string") {
    return data.message;
  }

  return JSON.stringify(data, null, 2);
}

/*
  Payload matches the current FastAPI backend.
*/
function createPayload(form) {
  return {
    age: numberValue(form.age),
    sex: numberValue(form.sex),

    "on thyroxine": numberValue(form["on thyroxine"]),
    "query on thyroxine": numberValue(
      form["query on thyroxine"]
    ),
    "on antithyroid medication": numberValue(
      form["on antithyroid medication"]
    ),

    sick: numberValue(form.sick),
    pregnant: numberValue(form.pregnant),

    "thyroid surgery": numberValue(
      form["thyroid surgery"]
    ),

    I131_treatment: numberValue(
      form["I131 treatment"]
    ),

    "query hypothyroid": numberValue(
      form["query hypothyroid"]
    ),
    "query hyperthyroid": numberValue(
      form["query hyperthyroid"]
    ),

    lithium: numberValue(form.lithium),
    goitre: numberValue(form.goitre),
    tumor: numberValue(form.tumor),
    hypopituitary: numberValue(form.hypopituitary),
    psych: numberValue(form.psych),

    TSH_measured: numberValue(form["TSH measured"]),
    TSH: numberValue(form.TSH),

    T3_measured: numberValue(form["T3 measured"]),

    TT4_measured: numberValue(form["TT4 measured"]),
    TT4: numberValue(form.TT4),

    T4U_measured: numberValue(form["T4U measured"]),
    T4U: numberValue(form.T4U),

    FTI_measured: numberValue(form["FTI measured"]),
    FTI: numberValue(form.FTI),
  };
}

function extractPrediction(data) {
  if (data?.prediction !== undefined) {
    return Number(data.prediction);
  }

  if (data?.predicted_class !== undefined) {
    return Number(data.predicted_class);
  }

  if (data?.class_prediction !== undefined) {
    return Number(data.class_prediction);
  }

  if (data?.result?.prediction !== undefined) {
    return Number(data.result.prediction);
  }

  return 0;
}

function extractModel(data) {
  return (
    data?.model ||
    data?.model_name ||
    data?.result?.model ||
    "XGBoost"
  );
}

function extractProbability(data, classNumber) {
  const directKey = `class_${classNumber}_probability`;

  if (data?.[directKey] !== undefined) {
    return Number(data[directKey]);
  }

  if (data?.probabilities?.[classNumber] !== undefined) {
    return Number(data.probabilities[classNumber]);
  }

  if (
    data?.probabilities?.[`class_${classNumber}`] !==
    undefined
  ) {
    return Number(
      data.probabilities[`class_${classNumber}`]
    );
  }

  return 0;
}

function extractShap(data) {
  const raw =
    data?.shap_values ||
    data?.shap ||
    data?.explanation ||
    data?.result?.shap_values ||
    [];

  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((item) => {
      if (Array.isArray(item)) {
        return {
          feature: item[0],
          value: Number(item[1]) || 0,
        };
      }

      return {
        feature:
          item?.feature ||
          item?.feature_name ||
          item?.name ||
          "Feature",
        value:
          Number(
            item?.value ??
              item?.shap_value ??
              item?.impact ??
              0
          ) || 0,
      };
    })
    .filter((item) => item.feature)
    .sort(
      (a, b) => Math.abs(b.value) - Math.abs(a.value)
    )
    .slice(0, 10);
}

function extractCounterfactuals(data) {
  const cf =
    data?.counterfactuals ||
    data?.counterfactual ||
    data?.result?.counterfactuals;

  if (!cf) {
    return {
      available: false,
      scenarios: [],
    };
  }

  return {
    available: Boolean(cf.available),
    scenarios: Array.isArray(cf.scenarios)
      ? cf.scenarios
      : [],
  };
}

function percentage(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "0.00%";
  }

  return `${(n <= 1 ? n * 100 : n).toFixed(2)}%`;
}

function displayNumber(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return String(value);
  }

  return Number.isInteger(n)
    ? String(n)
    : n.toFixed(4);
}

export default function App() {
  const [page, setPage] = useState("home");
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [openGroups, setOpenGroups] = useState({
    "Patient Information": true,
    "Medical History": true,
    "Thyroid Measurements": true,
  });

  const updateField = (field, value) => {
    setForm((old) => ({
      ...old,
      [field]: numericFields.includes(field)
        ? value
        : Number(value),
    }));
  };

  const resetForm = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  const toggleGroup = (title) => {
    setOpenGroups((old) => ({
      ...old,
      [title]: !old[title],
    }));
  };

  const predict = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 180000);

    try {
      const payload = createPayload(form);

      console.log("THYROCARE REQUEST:", payload);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        }
      );

      const responseText = await response.text();

      let data;

      try {
        data = responseText
          ? JSON.parse(responseText)
          : {};
      } catch {
        data = {
          detail:
            responseText ||
            "Invalid response received from server.",
        };
      }

      console.log("THYROCARE RESPONSE:", data);

      if (!response.ok) {
        throw new Error(
          formatApiError(data, response.status)
        );
      }

      const prediction = extractPrediction(data);

      setResult({
        prediction,
        model: extractModel(data),
        class0: extractProbability(data, 0),
        class1: extractProbability(data, 1),
        shap: extractShap(data),
        counterfactuals: extractCounterfactuals(data),
        raw: data,
      });

      setPage("results");
    } catch (err) {
      console.error("PREDICTION ERROR:", err);

      if (err?.name === "AbortError") {
        setError(
          "The prediction server took too long to respond.\n\n" +
            "Render may be waking the backend after inactivity. " +
            "Please wait a little and try again."
        );
      } else {
        setError(
          err?.message ||
            "Unable to connect to the prediction server."
        );
      }
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!result) return;

    const prediction =
      result.prediction === 1
        ? "Thyroid Disease Predicted"
        : "Thyroid Disease Not Predicted";

    let report = "";

    report += "THYROCARE - THYROID DIAGNOSIS AI\n";
    report += "====================================\n\n";
    report += `Prediction: ${prediction}\n`;
    report += `Model: ${result.model}\n\n`;

    report += "CLASS PROBABILITIES\n";
    report += "-------------------\n";
    report += `Class 0: ${percentage(result.class0)}\n`;
    report += `Class 1: ${percentage(result.class1)}\n\n`;

    report += "SHAP EXPLANATION\n";
    report += "----------------\n";

    if (result.shap.length === 0) {
      report += "No SHAP values available.\n";
    } else {
      result.shap.forEach((item, index) => {
        report += `${index + 1}. ${item.feature}: ${displayNumber(
          item.value
        )}\n`;
      });
    }

    report += "\nCOUNTERFACTUAL SCENARIOS\n";
    report += "------------------------\n";

    if (
      !result.counterfactuals.available ||
      result.counterfactuals.scenarios.length === 0
    ) {
      report += "No counterfactual scenarios available.\n";
    } else {
      result.counterfactuals.scenarios.forEach(
        (scenario) => {
          report += `Scenario ${scenario.scenario}\n`;

          scenario.changes?.forEach((change) => {
            report += `  ${change.feature}: ${change.original_value} -> ${change.counterfactual_value}\n`;
          });

          report += `  Prediction: ${
            scenario.prediction === 1
              ? "Thyroid Disease Predicted"
              : "Thyroid Disease Not Predicted"
          }\n`;

          report += `  Class 0: ${percentage(
            scenario.class_0_probability
          )}\n`;

          report += `  Class 1: ${percentage(
            scenario.class_1_probability
          )}\n\n`;
        }
      );
    }

    report += "DISCLAIMER\n";
    report +=
      "This result is for educational and research purposes only and is not a medical diagnosis.\n";

    const blob = new Blob([report], {
      type: "text/plain;charset=utf-8",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "thyrocare-report.txt";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const predictionText =
    result?.prediction === 1
      ? "Thyroid Disease Predicted"
      : "Thyroid Disease Not Predicted";

  return (
    <div className="app">

      {/* NAVBAR */}
      <header className="topbar">
        <div
          className="brand"
          onClick={() => setPage("home")}
        >
          <div className="brand-icon">
            <Brain size={25} />
          </div>

          <div>
            <div className="brand-name">
              ThyroCare
            </div>

            <div className="brand-subtitle">
              AI Thyroid Analysis
            </div>
          </div>
        </div>

        <nav className="navigation">
          <button
            className={page === "home" ? "nav-active" : ""}
            onClick={() => setPage("home")}
          >
            Home
          </button>

          <button
            className={
              page === "prediction"
                ? "nav-active"
                : ""
            }
            onClick={() => setPage("prediction")}
          >
            Prediction
          </button>

          <button
            className={
              page === "about" ? "nav-active" : ""
            }
            onClick={() => setPage("about")}
          >
            About
          </button>
        </nav>
      </header>

      <main>

        {/* HOME */}
        {page === "home" && (
          <section className="hero-section">
            <div className="hero-content">

              <div className="hero-badge">
                <Sparkles size={16} />
                AI-Powered Thyroid Analysis
              </div>

              <h1>
                Smarter Thyroid
                <span> Analysis with AI</span>
              </h1>

              <p>
                Explore thyroid disease prediction using
                machine learning, SHAP explainability,
                and counterfactual analysis.
              </p>

              <div className="hero-actions">
                <button
                  className="primary-button"
                  onClick={() =>
                    setPage("prediction")
                  }
                >
                  Start Prediction
                  <TrendingUp size={19} />
                </button>

                <button
                  className="secondary-button"
                  onClick={() => setPage("about")}
                >
                  Learn More
                </button>
              </div>

              <div className="hero-features">
                <div>
                  <ShieldCheck size={20} />
                  <span>Explainable AI</span>
                </div>

                <div>
                  <Brain size={20} />
                  <span>XGBoost</span>
                </div>

                <div>
                  <Activity size={20} />
                  <span>25 Features</span>
                </div>
              </div>
            </div>

            <div className="hero-visual">
              <div className="orb">
                <Brain size={95} />
              </div>

              <div className="floating-card card-one">
                <Activity size={20} />

                <div>
                  <strong>25</strong>
                  <span>Features</span>
                </div>
              </div>

              <div className="floating-card card-two">
                <ShieldCheck size={20} />

                <div>
                  <strong>AI</strong>
                  <span>Explainable</span>
                </div>
              </div>

              <div className="floating-card card-three">
                <Sparkles size={20} />

                <div>
                  <strong>XAI</strong>
                  <span>Insights</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* PREDICTION */}
        {page === "prediction" && (
          <section className="page-section">

            <div className="page-heading">
              <div className="section-badge">
                <Stethoscope size={18} />
                Prediction
              </div>

              <h2>
                Enter Patient Information
              </h2>

              <p>
                Enter the available patient and thyroid
                measurements.
              </p>
            </div>

            <div className="prediction-layout">

              <div className="form-container">

                {groups.map((group) => {
                  const isOpen =
                    openGroups[group.title];

                  return (
                    <div
                      className="form-group"
                      key={group.title}
                    >
                      <button
                        className="group-header"
                        onClick={() =>
                          toggleGroup(group.title)
                        }
                      >
                        <div className="group-title">
                          {group.icon}
                          <span>
                            {group.title}
                          </span>
                        </div>

                        {isOpen ? (
                          <ChevronUp size={20} />
                        ) : (
                          <ChevronDown size={20} />
                        )}
                      </button>

                      {isOpen && (
                        <div className="form-grid">
                          {group.fields.map(
                            (field) => {
                              const numeric =
                                numericFields.includes(
                                  field
                                );

                              return (
                                <div
                                  className="input-group"
                                  key={field}
                                >
                                  <label>
                                    {labels[field]}
                                  </label>

                                  {numeric ? (
                                    <input
                                      type="number"
                                      step={
                                        field === "age"
                                          ? "1"
                                          : "any"
                                      }
                                      value={
                                        form[field]
                                      }
                                      onChange={(e) =>
                                        updateField(
                                          field,
                                          e.target.value
                                        )
                                      }
                                    />
                                  ) : (
                                    <select
                                      value={
                                        form[field]
                                      }
                                      onChange={(e) =>
                                        updateField(
                                          field,
                                          e.target.value
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
                                  )}
                                </div>
                              );
                            }
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* ERROR */}
                {error && (
                  <div className="error-message">
                    <div className="error-title">
                      Prediction Error
                    </div>

                    <pre>{error}</pre>
                  </div>
                )}

                <div className="form-actions">

                  <button
                    className="secondary-button"
                    onClick={resetForm}
                    disabled={loading}
                  >
                    <RotateCcw size={18} />
                    Reset
                  </button>

                  <button
                    className="primary-button prediction-button"
                    onClick={predict}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Loader2
                          size={19}
                          className="spin"
                        />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles size={19} />
                        Analyze Thyroid
                      </>
                    )}
                  </button>

                </div>
              </div>

              <aside className="info-panel">

                <div className="info-icon">
                  <Info size={24} />
                </div>

                <h3>How it works</h3>

                <p>
                  The entered values are sent to the
                  machine-learning prediction API.
                </p>

                <div className="info-step">
                  <span>01</span>

                  <div>
                    <strong>Input</strong>
                    <p>
                      Patient and thyroid measurements
                    </p>
                  </div>
                </div>

                <div className="info-step">
                  <span>02</span>

                  <div>
                    <strong>Prediction</strong>
                    <p>
                      XGBoost analyzes the features
                    </p>
                  </div>
                </div>

                <div className="info-step">
                  <span>03</span>

                  <div>
                    <strong>Explanation</strong>
                    <p>
                      SHAP and counterfactual analysis
                    </p>
                  </div>
                </div>

              </aside>
            </div>
          </section>
        )}

        {/* RESULTS */}
        {page === "results" && result && (
          <section className="page-section results-section">

            <div className="page-heading">
              <div className="section-badge">
                <Sparkles size={18} />
                Analysis Complete
              </div>

              <h2>Prediction Results</h2>

              <p>
                Your machine-learning analysis has
                been completed.
              </p>
            </div>

            {/* MAIN RESULT */}
            <div
              className={`prediction-banner ${
                result.prediction === 1
                  ? "prediction-positive"
                  : "prediction-negative"
              }`}
            >
              <div className="prediction-banner-icon">
                {result.prediction === 1 ? (
                  <Activity size={34} />
                ) : (
                  <ShieldCheck size={34} />
                )}
              </div>

              <div>
                <span>Model Prediction</span>

                <h2>{predictionText}</h2>

                <p>
                  Model:{" "}
                  <strong>
                    {result.model}
                  </strong>
                </p>
              </div>
            </div>

            {/* PROBABILITIES */}
            <div className="result-grid">

              <div className="result-card">
                <span>
                  Class 0 Probability
                </span>

                <strong>
                  {percentage(result.class0)}
                </strong>

                <div className="progress">
                  <div
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          result.class0 <= 1
                            ? result.class0 * 100
                            : result.class0
                        )
                      )}%`,
                    }}
                  />
                </div>

                <small>
                  Thyroid Disease Not Predicted
                </small>
              </div>

              <div className="result-card">
                <span>
                  Class 1 Probability
                </span>

                <strong>
                  {percentage(result.class1)}
                </strong>

                <div className="progress">
                  <div
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          result.class1 <= 1
                            ? result.class1 * 100
                            : result.class1
                        )
                      )}%`,
                    }}
                  />
                </div>

                <small>
                  Thyroid Disease Predicted
                </small>
              </div>

            </div>

            {/* SHAP */}
            <div className="analysis-card">

              <div className="analysis-card-header">
                <div>
                  <div className="section-badge">
                    <Brain size={17} />
                    Explainable AI
                  </div>

                  <h3>
                    SHAP Feature Impact
                  </h3>
                </div>
              </div>

              {result.shap.length === 0 ? (
                <div className="empty-state">
                  SHAP explanation is not
                  available.
                </div>
              ) : (
                <div className="shap-list">

                  {result.shap.map(
                    (item, index) => (
                      <div
                        className="shap-item"
                        key={`${item.feature}-${index}`}
                      >
                        <div className="shap-number">
                          {index + 1}
                        </div>

                        <div className="shap-feature">
                          <span>
                            {item.feature}
                          </span>

                          <div className="shap-bar">
                            <div
                              style={{
                                width: `${Math.min(
                                  100,
                                  Math.abs(
                                    item.value
                                  ) * 100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>

                        <strong
                          className={
                            item.value >= 0
                              ? "impact-positive"
                              : "impact-negative"
                          }
                        >
                          {item.value >= 0
                            ? "+"
                            : ""}
                          {displayNumber(
                            item.value
                          )}
                        </strong>
                      </div>
                    )
                  )}

                </div>
              )}
            </div>

            {/* COUNTERFACTUAL */}
            <div className="analysis-card">

              <div className="analysis-card-header">
                <div>
                  <div className="section-badge">
                    <TrendingUp size={17} />
                    Counterfactual XAI
                  </div>

                  <h3>
                    What-if Scenarios
                  </h3>
                </div>
              </div>

              {!result.counterfactuals
                .available ||
              result.counterfactuals.scenarios
                .length === 0 ? (
                <div className="empty-state">
                  Counterfactual scenarios are
                  not available.
                </div>
              ) : (
                <div className="counterfactual-list">

                  {result.counterfactuals.scenarios.map(
                    (scenario) => (
                      <div
                        className="counterfactual-card"
                        key={
                          scenario.scenario
                        }
                      >
                        <div className="scenario-header">
                          <span>
                            Scenario{" "}
                            {scenario.scenario}
                          </span>

                          <strong>
                            {scenario.prediction ===
                            1
                              ? "Disease Predicted"
                              : "Not Predicted"}
                          </strong>
                        </div>

                        <div className="scenario-changes">

                          {scenario.changes?.map(
                            (
                              change,
                              index
                            ) => (
                              <div
                                className="change-row"
                                key={index}
                              >
                                <span>
                                  {
                                    change.feature
                                  }
                                </span>

                                <div>
                                  <b>
                                    {displayNumber(
                                      change.original_value
                                    )}
                                  </b>

                                  <span className="arrow">
                                    →
                                  </span>

                                  <b>
                                    {displayNumber(
                                      change.counterfactual_value
                                    )}
                                  </b>
                                </div>
                              </div>
                            )
                          )}

                        </div>

                        <div className="scenario-probabilities">

                          <div>
                            <span>
                              Class 0
                            </span>

                            <strong>
                              {percentage(
                                scenario.class_0_probability
                              )}
                            </strong>
                          </div>

                          <div>
                            <span>
                              Class 1
                            </span>

                            <strong>
                              {percentage(
                                scenario.class_1_probability
                              )}
                            </strong>
                          </div>

                        </div>
                      </div>
                    )
                  )}

                </div>
              )}

              <div className="counterfactual-note">
                <Info size={17} />

                <span>
                  Counterfactual scenarios are
                  hypothetical machine-learning
                  sensitivity scenarios. They are not
                  medical recommendations.
                </span>
              </div>

            </div>

            {/* ACTIONS */}
            <div className="result-actions">

              <button
                className="secondary-button"
                onClick={() =>
                  setPage("prediction")
                }
              >
                <RotateCcw size={18} />
                New Prediction
              </button>

              <button
                className="primary-button"
                onClick={downloadReport}
              >
                <Download size={18} />
                Download Report
              </button>

            </div>

            <div className="medical-disclaimer">
              <ShieldCheck size={20} />

              <p>
                <strong>Important:</strong>{" "}
                This application is intended for
                educational and research purposes.
                The prediction is not a medical
                diagnosis and should not replace
                advice from a qualified healthcare
                professional.
              </p>
            </div>

          </section>
        )}

        {/* ABOUT */}
        {page === "about" && (
          <section className="page-section">

            <div className="page-heading">
              <div className="section-badge">
                <Brain size={18} />
                About ThyroCare
              </div>

              <h2>
                Machine Learning + Explainable AI
              </h2>

              <p>
                ThyroCare demonstrates the use of
                machine learning and explainable AI
                techniques for thyroid disease
                analysis.
              </p>
            </div>

            <div className="about-grid">

              <div className="about-card">
                <Brain size={30} />

                <h3>XGBoost</h3>

                <p>
                  The application uses an XGBoost
                  classifier trained using thyroid
                  related clinical features.
                </p>
              </div>

              <div className="about-card">
                <Sparkles size={30} />

                <h3>SHAP</h3>

                <p>
                  SHAP provides feature-level
                  explanations showing how input
                  features influence model output.
                </p>
              </div>

              <div className="about-card">
                <TrendingUp size={30} />

                <h3>Counterfactual XAI</h3>

                <p>
                  Counterfactual scenarios show
                  hypothetical changes that can
                  influence the model prediction.
                </p>
              </div>

              <div className="about-card">
                <ShieldCheck size={30} />

                <h3>Academic Project</h3>

                <p>
                  This system is designed for
                  academic, demonstration, and
                  research purposes.
                </p>
              </div>

            </div>
          </section>
        )}

      </main>

      <footer className="footer">
        <div>
          <strong>ThyroCare</strong>
          <span>
            {" "}
            • AI Thyroid Analysis
          </span>
        </div>

        <div>
          Final Year B.Tech Project
        </div>
      </footer>

    </div>
  );
    }
