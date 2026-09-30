import React, { useMemo, useState } from "react";
import "./index.css";

import AdminLogin from "./admin/AdminLogin";
import AdminDashboard from "./admin/AdminDashboard";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

const ADMIN_USERNAME = "admin";
const ADMIN_PASSWORD = "thyrocare123";

/* =========================================================
   INITIAL FORM
========================================================= */

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

/* =========================================================
   FORM GROUPS
========================================================= */

const clinicalFields = [
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
];

const measurementAvailability = [
  "TSH measured",
  "T3 measured",
  "TT4 measured",
  "T4U measured",
  "FTI measured",
];

const thyroidMeasurements = [
  {
    key: "TSH",
    label: "TSH",
    description: "Thyroid-stimulating hormone",
  },
  {
    key: "TT4",
    label: "TT4",
    description: "Total thyroxine",
  },
  {
    key: "T4U",
    label: "T4U",
    description: "T4 uptake",
  },
  {
    key: "FTI",
    label: "FTI",
    description: "Free thyroxine index",
  },
];

/* =========================================================
   HELPERS
========================================================= */

function formatNumber(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "0.00";
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

/* =========================================================
   SHAP NORMALIZER
========================================================= */

function normalizeShap(shap) {
  if (!shap) {
    return [];
  }

  if (
    typeof shap === "object" &&
    !Array.isArray(shap) &&
    Array.isArray(shap.features)
  ) {
    return shap.features.map((item) => ({
      feature:
        item?.feature ||
        item?.name ||
        item?.column ||
        "Feature",

      value: Number(
        item?.value ??
          item?.shap_value ??
          item?.impact ??
          0
      ),
    }));
  }

  if (Array.isArray(shap)) {
    return shap.map((item) => {
      if (
        typeof item === "object" &&
        item !== null
      ) {
        return {
          feature:
            item.feature ||
            item.name ||
            item.column ||
            "Feature",

          value: Number(
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
    return Object.entries(shap).map(
      ([feature, value]) => ({
        feature,
        value: Number(value) || 0,
      })
    );
  }

  return [];
}

/* =========================================================
   COUNTERFACTUAL NORMALIZER
========================================================= */

function normalizeCounterfactuals(counterfactuals) {
  if (!counterfactuals) {
    return [];
  }

  /*
   * CURRENT BACKEND FORMAT
   *
   * {
   *   available: true,
   *   features: [...],
   *   scenarios: [
   *     {
   *       changes: [
   *         {
   *           feature: "TSH",
   *           original_value: 0,
   *           counterfactual_value: 7
   *         }
   *       ],
   *       prediction: 1,
   *       prediction_label: "...",
   *       class_0_probability: ...,
   *       class_1_probability: ...,
   *       distance: ...
   *     }
   *   ]
   * }
   */

  if (
    typeof counterfactuals === "object" &&
    !Array.isArray(counterfactuals) &&
    Array.isArray(counterfactuals.scenarios)
  ) {
    return counterfactuals.scenarios.map(
      (scenario, index) => {
        const changes = Array.isArray(
          scenario?.changes
        )
          ? scenario.changes
          : [];

        const change = changes[0] || {};

        const prediction =
          scenario?.prediction;

        let probability = null;

        if (prediction === 1) {
          probability =
            scenario?.class_1_probability ??
            null;
        }

        if (prediction === 0) {
          probability =
            scenario?.class_0_probability ??
            null;
        }

        return {
          id: index,

          feature:
            change?.feature ||
            "Feature",

          from:
            change?.original_value ??
            "-",

          to:
            change?.counterfactual_value ??
            "-",

          prediction:
            scenario?.prediction_label ??
            prediction ??
            "-",

          probability,

          distance:
            scenario?.distance ??
            null,
        };
      }
    );
  }

  /*
   * If backend wraps object inside data.
   */

  if (
    typeof counterfactuals === "object" &&
    !Array.isArray(counterfactuals) &&
    Array.isArray(counterfactuals.data)
  ) {
    return normalizeCounterfactuals(
      counterfactuals.data
    );
  }

  /*
   * Direct scenario property.
   */

  if (
    typeof counterfactuals === "object" &&
    !Array.isArray(counterfactuals) &&
    Array.isArray(counterfactuals.scenario)
  ) {
    return normalizeCounterfactuals({
      scenarios:
        counterfactuals.scenario,
    });
  }

  /*
   * Array format.
   */

  if (Array.isArray(counterfactuals)) {
    return counterfactuals.map(
      (item, index) => {
        const changes = Array.isArray(
          item?.changes
        )
          ? item.changes
          : [];

        const change =
          changes[0] || null;

        if (change) {
          const prediction =
            item?.prediction;

          let probability = null;

          if (prediction === 1) {
            probability =
              item?.class_1_probability ??
              null;
          }

          if (prediction === 0) {
            probability =
              item?.class_0_probability ??
              null;
          }

          return {
            id: index,

            feature:
              change?.feature ||
              "Feature",

            from:
              change?.original_value ??
              "-",

            to:
              change?.counterfactual_value ??
              "-",

            prediction:
              item?.prediction_label ??
              prediction ??
              "-",

            probability,

            distance:
              item?.distance ??
              null,
          };
        }

        return {
          id: index,

          feature:
            item?.feature ||
            item?.changed_feature ||
            "Feature",

          from:
            item?.from ??
            item?.original ??
            item?.old_value ??
            item?.original_value ??
            "-",

          to:
            item?.to ??
            item?.new_value ??
            item?.changed_to ??
            item?.counterfactual_value ??
            "-",

          prediction:
            item?.prediction_label ??
            item?.prediction ??
            item?.class ??
            item?.target ??
            "-",

          probability:
            item?.probability ??
            item?.class_probability ??
            item?.prob ??
            null,

          distance:
            item?.distance ??
            null,
        };
      }
    );
  }

  return [];
}

/* =========================================================
   MAIN APP
========================================================= */

export default function App() {
  const [page, setPage] =
    useState("home");

  const [form, setForm] =
    useState({
      ...initialForm,
    });

  /*
   * IMPORTANT:
   * Restore the latest result when the
   * application is opened again during
   * the same browser session.
   */

  const [result, setResult] =
    useState(() => {
      try {
        const saved =
          sessionStorage.getItem(
            "thyrocare_latest_result"
          );

        return saved
          ? JSON.parse(saved)
          : null;
      } catch {
        return null;
      }
    });

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [menuOpen, setMenuOpen] =
    useState(false);

  /* =======================================================
     ADMIN AUTH
  ======================================================= */

  const [
    adminAuthenticated,
    setAdminAuthenticated,
  ] = useState(() => {
    try {
      return (
        sessionStorage.getItem(
          "thyrocare_admin"
        ) === "true"
      );
    } catch {
      return false;
    }
  });

  /* =======================================================
     HISTORY
  ======================================================= */

  /*
   * IMPORTANT:
   * Restore prediction history from
   * sessionStorage.
   */

  const [
    analysisHistory,
    setAnalysisHistory,
  ] = useState(() => {
    try {
      const saved =
        sessionStorage.getItem(
          "thyrocare_analysis_history"
        );

      return saved
        ? JSON.parse(saved)
        : [];
    } catch {
      return [];
    }
  });

  /* =======================================================
     NAVIGATION
  ======================================================= */

  const goTo = (nextPage) => {
    if (
      nextPage === "admin" &&
      !adminAuthenticated
    ) {
      setPage("admin-login");
    } else {
      setPage(nextPage);
    }

    setMenuOpen(false);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  /* =======================================================
     ADMIN LOGIN
  ======================================================= */

  const handleAdminLogin = (
    username,
    password
  ) => {
    const valid =
      username === ADMIN_USERNAME &&
      password === ADMIN_PASSWORD;

    if (!valid) {
      return false;
    }

    setAdminAuthenticated(true);

    try {
      sessionStorage.setItem(
        "thyrocare_admin",
        "true"
      );
    } catch {}

    setPage("admin");
    setMenuOpen(false);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });

    return true;
  };

  /* =======================================================
     ADMIN LOGOUT
  ======================================================= */

  const handleAdminLogout = () => {
    setAdminAuthenticated(false);

    try {
      sessionStorage.removeItem(
        "thyrocare_admin"
      );
    } catch {}

    setPage("home");
    setMenuOpen(false);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  /* =======================================================
     UPDATE FIELD
  ======================================================= */

  const updateField = (
    field,
    value
  ) => {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  /* =======================================================
     BUILD API PAYLOAD
  ======================================================= */

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

      on_antithyroid_medication:
        Number(
          form[
            "on antithyroid medication"
          ]
        ),

      sick: Number(form.sick),

      pregnant: Number(
        form.pregnant
      ),

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

      lithium: Number(
        form.lithium
      ),

      goitre: Number(
        form.goitre
      ),

      tumor: Number(
        form.tumor
      ),

      hypopituitary: Number(
        form.hypopituitary
      ),

      psych: Number(
        form.psych
      ),

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

  /* =======================================================
     ANALYZE
  ======================================================= */

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");

    let timeout;

    try {
      const controller =
        new AbortController();

      timeout = setTimeout(() => {
        controller.abort();
      }, 180000);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify(
            buildPayload()
          ),

          signal: controller.signal,
        }
      );

      clearTimeout(timeout);

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The prediction server returned an invalid response."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            "Prediction request failed."
        );
      }

      /* =================================================
         SAVE COMPLETE RESULT
      ================================================= */

      setResult(data);

      /*
       * Persist latest result so the
       * Admin Dashboard can still access
       * it after navigation.
       */

      try {
        sessionStorage.setItem(
          "thyrocare_latest_result",
          JSON.stringify(data)
        );
      } catch {}


      /* =================================================
         EXTRACT PREDICTION
      ================================================= */

      const predicted =
        data?.prediction ??
        data?.predicted_class ??
        data?.class ??
        null;


      /* =================================================
         EXTRACT PROBABILITIES
      ================================================= */

      const probabilities =
        data?.probabilities ||
        data?.class_probabilities ||
        {};

      const historyClass0 =
        probabilities?.["0"] ??
        probabilities?.class_0 ??
        probabilities?.not_disease ??
        data?.class_0_probability ??
        0;

      const historyClass1 =
        probabilities?.["1"] ??
        probabilities?.class_1 ??
        probabilities?.disease ??
        data?.class_1_probability ??
        0;


      /* =================================================
         MODEL NAME
      ================================================= */

      const modelName =
        data?.model_name ||
        data?.model ||
        data?.algorithm ||
        "XGBoost";


      /* =================================================
         CREATE HISTORY ITEM
      ================================================= */

      const historyItem = {
        id: Date.now(),

        time:
          new Date().toISOString(),

        prediction: predicted,

        probabilities: {
          "0": Number(
            historyClass0
          ),

          "1": Number(
            historyClass1
          ),
        },

        model: modelName,
      };


      /* =================================================
         SAVE HISTORY
      ================================================= */

      setAnalysisHistory(
        (previous) => {
          const nextHistory = [
            historyItem,
            ...previous,
          ].slice(0, 20);

          try {
            sessionStorage.setItem(
              "thyrocare_analysis_history",
              JSON.stringify(
                nextHistory
              )
            );
          } catch {}

          return nextHistory;
        }
      );


      /* =================================================
         GO TO RESULTS
      ================================================= */

      setPage("results");

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });

    } catch (err) {
      if (
        err?.name ===
        "AbortError"
      ) {
        setError(
          "Prediction request timed out. Please try again."
        );
      } else {
        setError(
          err?.message ||
            "Unable to connect to prediction server."
        );
      }
    } finally {
      if (timeout) {
        clearTimeout(timeout);
      }

      setLoading(false);
    }
  };

  /* =======================================================
     RESET
  ======================================================= */

  const resetForm = () => {
    setForm({
      ...initialForm,
    });

    setResult(null);

    /*
     * Clear only the latest result.
     *
     * Prediction history is preserved.
     */

    try {
      sessionStorage.removeItem(
        "thyrocare_latest_result"
      );
    } catch {}

    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  /* =======================================================
     SHAP DATA
  ======================================================= */

  const shapData = useMemo(() => {
    return normalizeShap(
      result?.shap_values ||
        result?.shap ||
        result?.explanation
    );
  }, [result]);

  /* =======================================================
     COUNTERFACTUAL DATA
  ======================================================= */

  const counterfactualData =
    useMemo(() => {
      const source =
        result?.counterfactuals ??
        result?.counterfactual_explanations ??
        result?.counterfactual ??
        result?.counterfactual_data ??
        null;

      return normalizeCounterfactuals(
        source
      );
    }, [result]);

  /* =======================================================
     PREDICTION
  ======================================================= */

  const predictionClass =
    result?.prediction ??
    result?.predicted_class ??
    result?.class ??
    null;

  /* =======================================================
     PROBABILITIES
  ======================================================= */

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

  /* =======================================================
     LABEL
  ======================================================= */

  const predictionLabel =
    predictionClass === 1
      ? "Thyroid Disease Predicted"
      : predictionClass === 0
      ? "Thyroid Disease Not Predicted"
      : "Prediction Result";

  /* =======================================================
     NAVBAR
  ======================================================= */

  const renderNavbar = () => {
    return (
      <header className="site-header">
        <nav className="navbar">

          <button
            className="brand"
            onClick={() =>
              goTo("home")
            }
          >
            <span className="brand-mark">
              TC
            </span>

            <span className="brand-text">
              <strong>
                ThyroCare<span>AI</span>
              </strong>

              <small>
                THYROID ANALYSIS
              </small>
            </span>
          </button>

          <div
            className={`nav-links ${
              menuOpen ? "open" : ""
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
                page === "admin-login"
                  ? "active"
                  : ""
              }
              onClick={() =>
                goTo("admin")
              }
            >
              {adminAuthenticated
                ? "Dashboard"
                : "Admin"}
            </button>
          </div>

          <button
            className={`mobile-menu-button ${
              menuOpen ? "active" : ""
            }`}
            onClick={() =>
              setMenuOpen(
                (previous) =>
                  !previous
              )
            }
          >
            <span />
            <span />
            <span />
          </button>

        </nav>
      </header>
    );
  };

  /* =======================================================
     HOME
  ======================================================= */

  const renderHomePage = () => {
    return (
      <main className="home-page">

        <section className="hero">

          <div className="hero-content">

            <div className="hero-badge">
              <span className="status-dot" />
              AI-POWERED THYROID ANALYSIS
            </div>

            <h1>
              Smarter
              <br />
              <span>
                Thyroid Analysis.
              </span>
            </h1>

            <p className="hero-description">
              An AI-powered thyroid disease
              prediction system using
              XGBoost, SHAP explainability
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

            <div className="hero-note">
              <span>●</span>
              25 clinical features ·
              XGBoost · Explainable AI
            </div>

          </div>

          <div className="hero-visual">

            <div className="medical-orb">

              <div className="orb-ring ring-one" />
              <div className="orb-ring ring-two" />
              <div className="orb-ring ring-three" />

              <div className="orb-core">
                <span>AI</span>
                <small>
                  THYROID
                </small>
              </div>

              <div className="floating-card card-one">
                <span>
                  MODEL
                </span>

                <strong>
                  XGBoost
                </strong>
              </div>

              <div className="floating-card card-two">
                <span>
                  ACCURACY
                </span>

                <strong>
                  99.87%
                </strong>
              </div>

              <div className="floating-card card-three">
                <span>
                  XAI
                </span>

                <strong>
                  SHAP + CF
                </strong>
              </div>

            </div>

          </div>

        </section>

        <section className="stats-section">

          <div className="stat-card">
            <span className="stat-number">
              99.87%
            </span>

            <span className="stat-label">
              Test Accuracy
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-number">
              25
            </span>

            <span className="stat-label">
              Clinical Features
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-number">
              XGB
            </span>

            <span className="stat-label">
              Prediction Model
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-number">
              XAI
            </span>

            <span className="stat-label">
              Explainable AI
            </span>
          </div>

        </section>

        <section className="workflow-section">

          <div className="section-heading">

            <span>
              HOW IT WORKS
            </span>

            <h2>
              From patient data
              <br />
              to explainable prediction.
            </h2>

            <p>
              The system combines machine
              learning with explainable AI
              to make the prediction process
              easier to understand.
            </p>

          </div>

          <div className="workflow-grid">

            <div className="workflow-card">

              <span className="step-number">
                01
              </span>

              <div className="workflow-icon">
                ↓
              </div>

              <h3>
                Input
              </h3>

              <p>
                Enter patient and thyroid
                measurement values through
                a structured interface.
              </p>

            </div>

            <div className="workflow-card">

              <span className="step-number">
                02
              </span>

              <div className="workflow-icon">
                ◈
              </div>

              <h3>
                Prediction
              </h3>

              <p>
                The trained XGBoost model
                analyzes the 25 clinical
                features.
              </p>

            </div>

            <div className="workflow-card">

              <span className="step-number">
                03
              </span>

              <div className="workflow-icon">
                ✦
              </div>

              <h3>
                Explanation
              </h3>

              <p>
                SHAP identifies feature
                contributions while
                counterfactual analysis
                provides what-if scenarios.
              </p>

            </div>

          </div>

        </section>

        <section className="cta-section">

          <div>

            <span>
              READY TO ANALYZE?
            </span>

            <h2>
              Explore your thyroid
              analysis.
            </h2>

          </div>

          <button
            className="primary-button"
            onClick={() =>
              goTo("prediction")
            }
          >
            Start Analysis →
          </button>

        </section>

      </main>
    );
  };

  /* =======================================================
     PREDICTION PAGE
  ======================================================= */

  const renderPredictionPage = () => {
    return (
      <main className="prediction-page">

        <section className="page-hero">

          <div>

            <span>
              AI THYROID ANALYSIS
            </span>

            <h1>
              Analyze Thyroid
            </h1>

            <p>
              Enter patient measurements
              and clinical indicators
              to generate an AI-powered
              prediction.
            </p>

          </div>

          <div className="model-badge">

            <small>
              MODEL
            </small>

            <strong>
              XGBoost
            </strong>

            <span>
              99.87% accuracy
            </span>

          </div>

        </section>

        {error && (
          <div className="error-box">

            <strong>
              Analysis Error
            </strong>

            <span>
              {error}
            </span>

          </div>
        )}

        <section className="form-card">

          <div className="form-section">

            <div className="form-section-heading">

              <span>
                01
              </span>

              <div>
                <h2>
                  Patient Information
                </h2>

                <p>
                  Basic patient
                  information.
                </p>
              </div>

            </div>

            <div className="form-grid">

              <div className="input-group">

                <label>
                  Age
                </label>

                <input
                  type="number"
                  min="0"
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

                <label>
                  Sex
                </label>

                <select
                  value={form.sex}
                  onChange={(e) =>
                    updateField(
                      "sex",
                      Number(
                        e.target.value
                      )
                    )
                  }
                >
                  <option value={0}>
                    Female / 0
                  </option>

                  <option value={1}>
                    Male / 1
                  </option>
                </select>

              </div>

            </div>

          </div>

          <div className="form-section">

            <div className="form-section-heading">

              <span>
                02
              </span>

              <div>
                <h2>
                  Thyroid Measurements
                </h2>

                <p>
                  Enter the available
                  laboratory measurements.
                </p>
              </div>

            </div>

            <div className="form-grid">

              {thyroidMeasurements.map(
                (item) => (
                  <div
                    className="input-group"
                    key={item.key}
                  >

                    <label>
                      {item.label}
                    </label>

                    <small>
                      {item.description}
                    </small>

                    <input
                      type="number"
                      step="any"
                      value={
                        form[item.key]
                      }
                      onChange={(e) =>
                        updateField(
                          item.key,
                          e.target.value
                        )
                      }
                    />

                  </div>
                )
              )}

            </div>

          </div>

          <div className="form-section">

            <div className="form-section-heading">

              <span>
                03
              </span>

              <div>
                <h2>
                  Clinical Indicators
                </h2>

                <p>
                  Select the patient's
                  clinical indicators.
                </p>

              </div>

            </div>

            <div className="binary-grid">

              {clinicalFields.map(
                (field) => (
                  <div
                    className="select-field"
                    key={field}
                  >

                    <label>
                      {field}
                    </label>

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
                        No / 0
                      </option>

                      <option value={1}>
                        Yes / 1
                      </option>

                    </select>

                  </div>
                )
              )}

            </div>

          </div>

          <div className="form-section">

            <div className="form-section-heading">

              <span>
                04
              </span>

              <div>
                <h2>
                  Measurement Availability
                </h2>

                <p>
                  Indicate whether each
                  measurement is available.
                </p>
              </div>

            </div>

            <div className="binary-grid">

              {measurementAvailability.map(
                (field) => (
                  <div
                    className="select-field"
                    key={field}
                  >

                    <label>
                      {field}
                    </label>

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
                        No / 0
                      </option>

                      <option value={1}>
                        Yes / 1
                      </option>

                    </select>

                  </div>
                )
              )}

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
              onClick={
                handleAnalyze
              }
              disabled={loading}
            >

              {loading ? (
                <>
                  <span className="spinner" />
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

          </div>

        </section>

      </main>
    );
  };

  /* =======================================================
     RESULTS PAGE
  ======================================================= */

  const renderResultsPage = () => {
    if (!result) {
      return (
        <main className="empty-results">

          <div className="empty-results-icon">
            ◌
          </div>

          <span>
            RESULTS
          </span>

          <h1>
            No Analysis Yet
          </h1>

          <p>
            Run a thyroid analysis to
            view the prediction and
            explainable AI insights.
          </p>

          <button
            className="primary-button"
            onClick={() =>
              goTo("prediction")
            }
          >
            Start Analysis →
          </button>

        </main>
      );
    }

    return (
      <main className="results-page">

        <section className="page-hero">

          <div>

            <span>
              ANALYSIS COMPLETE
            </span>

            <h1>
              Prediction Results
            </h1>

            <p>
              Machine learning prediction
              with explainable AI insights.
            </p>

          </div>

          <div className="model-badge">

            <small>
              MODEL
            </small>

            <strong>
              {result.model ||
                result.model_name ||
                "XGBoost"}
            </strong>

          </div>

        </section>

        <section
          className={`prediction-banner ${
            predictionClass === 1
              ? "positive"
              : "negative"
          }`}
        >

          <div className="prediction-icon">
            {predictionClass === 1
              ? "!"
              : "✓"}
          </div>

          <div>

            <span>
              MODEL PREDICTION
            </span>

            <h2>
              {predictionLabel}
            </h2>

            <p>
              XGBoost clinical
              classification result
            </p>

          </div>

        </section>

        <section className="probability-section">

          <div className="section-heading left">

            <span>
              PROBABILITY
            </span>

            <h2>
              Prediction Confidence
            </h2>

          </div>

          <div className="probability-grid">

            <div className="probability-card">

              <div>

                <span>
                  CLASS 0
                </span>

                <strong>
                  {formatPercent(
                    class0
                  )}
                </strong>

              </div>

              <div className="progress-track">

                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(
                        100,
                        Number(
                          class0
                        ) * 100
                      )
                    )}%`,
                  }}
                />

              </div>

              <p>
                Thyroid Disease Not
                Predicted
              </p>

            </div>

            <div className="probability-card">

              <div>

                <span>
                  CLASS 1
                </span>

                <strong>
                  {formatPercent(
                    class1
                  )}
                </strong>

              </div>

              <div className="progress-track">

                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(
                        100,
                        Number(
                          class1
                        ) * 100
                      )
                    )}%`,
                  }}

                />

              </div>

              <p>
                Thyroid Disease
                Predicted
              </p>

            </div>

          </div>

        </section>

        {/* =================================================
            SHAP
        ================================================= */}

        <section className="explanation-section">

          <div className="section-heading left">

            <span>
              01 · EXPLAINABLE AI
            </span>

            <h2>
              SHAP Feature Impact
            </h2>

            <p>
              SHAP values show how
              individual features
              contributed to the model
              prediction.
            </p>

          </div>

          {shapData.length > 0 ? (

            <div className="shap-list">

              {shapData
                .slice(0, 10)
                .map(
                  (
                    item,
                    index
                  ) => (

                    <div
                      className="shap-row"
                      key={`${item.feature}-${index}`}
                    >

                      <div className="shap-name">

                        <span>
                          {String(
                            index + 1
                          ).padStart(
                            2,
                            "0"
                          )}
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
                            item.value >=
                            0
                              ? "positive-bar"
                              : "negative-bar"
                          }
                          style={{
                            width: `${Math.min(
                              100,
                              Math.abs(
                                Number(
                                  item.value
                                )
                              ) * 100
                            )}%`,
                          }}
                        />

                      </div>

                      <div
                        className={`shap-value ${
                          item.value >=
                          0
                            ? "positive-text"
                            : "negative-text"
                        }`}
                      >

                        {item.value >=
                        0
                          ? "+"
                          : ""}

                        {formatNumber(
                          item.value
                        )}

                      </div>

                    </div>

                  )
                )}

            </div>

          ) : (

            <div className="empty-card">
              SHAP explanation is not
              available for this prediction.
            </div>

          )}

        </section>

        {/* =================================================
            COUNTERFACTUAL
        ================================================= */}

        <section className="explanation-section">

          <div className="section-heading left">

            <span>
              02 · COUNTERFACTUAL AI
            </span>

            <h2>
              What Could Change the
              Prediction?
            </h2>

            <p>
              Counterfactual examples
              demonstrate how changing
              selected input values may
              affect the model output.
            </p>

          </div>

          {counterfactualData.length > 0 ? (

            <div className="counterfactual-grid">

              {counterfactualData.map(
                (
                  item,
                  index
                ) => (

                  <div
                    className="counterfactual-card"
                    key={
                      item.id ??
                      index
                    }
                  >

                    <span>
                      {item.feature}
                    </span>

                    <div className="cf-values">

                      <strong>
                        {formatNumber(
                          item.from
                        )}
                      </strong>

                      <span>
                        →
                      </span>

                      <strong>
                        {formatNumber(
                          item.to
                        )}
                      </strong>

                    </div>

                    <small>
                      Prediction:{" "}
                      {
                        item.prediction
                      }
                    </small>

                    {item.probability !==
                      null &&
                      Number.isFinite(
                        Number(
                          item.probability
                        )
                      ) && (
                        <small>
                          Probability:{" "}
                          {formatPercent(
                            item.probability
                          )}
                        </small>
                      )}

                    {item.distance !==
                      null &&
                      Number.isFinite(
                        Number(
                          item.distance
                        )
                      ) && (
                        <small>
                          Change distance:{" "}
                          {formatNumber(
                            item.distance
                          )}
                        </small>
                      )}

                  </div>
                )
              )}

            </div>

          ) : (

            <div className="empty-card">

              No counterfactual examples
              available.

            </div>

          )}

        </section>

        {/* =================================================
            RESULT ACTIONS
        ================================================= */}

        <section className="result-actions">

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
================================

Prediction:
${predictionLabel}

Class 0 Probability:
${formatPercent(class0)}

Class 1 Probability:
${formatPercent(class1)}

Model:
${
  result.model ||
  result.model_name ||
  "XGBoost"
}

SHAP FEATURES
================================

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
================================

${counterfactualData
  .map(
    (item) =>
      `${item.feature}: ${item.from} -> ${item.to} | ${item.prediction}`
  )
  .join("\n")}

================================

ThyroCare AI
Final Year B.Tech Project
Machine Learning & Explainable AI

DISCLAIMER:
This system is an academic AI project
and is not a medical diagnosis.
Consult a qualified healthcare
professional for medical decisions.
`;

              const blob =
                new Blob(
                  [report],
                  {
                    type: "text/plain",
                  }
                );

              const url =
                URL.createObjectURL(
                  blob
                );

              const link =
                document.createElement(
                  "a"
                );

              link.href = url;

              link.download =
                "thyrocare-ai-report.txt";

              document.body.appendChild(
                link
              );

              link.click();

              link.remove();

              URL.revokeObjectURL(
                url
              );
            }}
          >
            Download Report ↓
          </button>

        </section>

      </main>
    );
  };

  /* =======================================================
     ABOUT
  ======================================================= */

  const renderAboutPage = () => {
    return (
      <main className="about-page">

        <section className="page-hero about-hero">

          <div>

            <span>
              ABOUT THE PROJECT
            </span>

            <h1>
              ThyroCare AI
            </h1>

            <p>
              Enhancing Thyroid Disease
              Diagnosis With Machine
              Learning and Counterfactual
              Explainable AI.
            </p>

          </div>

        </section>

        <section className="about-grid">

          <div className="about-card">

            <div className="about-icon">
              AI
            </div>

            <span>
              01
            </span>

            <h2>
              Machine Learning
            </h2>

            <p>
              The system uses an XGBoost
              machine learning model
              trained on thyroid clinical
              data to classify thyroid
              disease.
            </p>

          </div>

          <div className="about-card">

            <div className="about-icon">
              S
            </div>

            <span>
              02
            </span>

            <h2>
              SHAP Explainability
            </h2>

            <p>
              SHAP values provide insight
              into which clinical features
              influence the model's output.
            </p>

          </div>

          <div className="about-card">

            <div className="about-icon">
              CF
            </div>

            <span>
              03
            </span>

            <h2>
              Counterfactual Analysis
            </h2>

            <p>
              Counterfactual examples
              demonstrate how changing
              selected measurements can
              affect model predictions.
            </p>

          </div>

        </section>

        <section className="technology-card">

          <span>
            TECHNOLOGY
          </span>

          <h2>
            Built with modern AI
            technologies.
          </h2>

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

        </section>

      </main>
    );
  };

  /* =======================================================
     FOOTER
  ======================================================= */

  const renderFooter = () => {
    return (
      <footer className="footer">

        <div className="footer-brand">

          <strong>
            ThyroCare<span>AI</span>
          </strong>

          <span>
            Thyroid Analysis
          </span>

        </div>

        <p>
          Final Year B.Tech Project ·
          Machine Learning & Explainable AI
        </p>

      </footer>
    );
  };

  /* =======================================================
     ADMIN LOGIN
  ======================================================= */

  if (
    page === "admin-login"
  ) {
    return (
      <div className="app">

        <AdminLogin
          onLogin={
            handleAdminLogin
          }
          onBack={() =>
            goTo("home")
          }
        />

      </div>
    );
  }

  /* =======================================================
     ADMIN DASHBOARD
  ======================================================= */

  if (page === "admin") {
    if (!adminAuthenticated) {
      return (
        <div className="app">

          <AdminLogin
            onLogin={
              handleAdminLogin
            }
            onBack={() =>
              goTo("home")
            }
          />

        </div>
      );
    }

    return (
      <div className="app">

        <AdminDashboard
          result={result}
          analysisHistory={
            analysisHistory
          }
          onNewAnalysis={() =>
            goTo("prediction")
          }
          onViewResults={() =>
            goTo("results")
          }
          onLogout={
            handleAdminLogout
          }
        />

      </div>
    );
  }

  /* =======================================================
     PUBLIC WEBSITE
  ======================================================= */

  return (
    <div className="app">

      {renderNavbar()}

      {page === "home" &&
        renderHomePage()}

      {page === "prediction" &&
        renderPredictionPage()}

      {page === "results" &&
        renderResultsPage()}

      {page === "about" &&
        renderAboutPage()}

      {renderFooter()}

    </div>
  );
         }
