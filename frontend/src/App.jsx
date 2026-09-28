import { useState } from "react";
import axios from "axios";
import {
  Activity,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  FileText,
  HeartPulse,
  Home,
  Info,
  Loader2,
  Menu,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  X,
} from "lucide-react";

const API_URL = "https://thyrocare-diagnosis-ai.onrender.com";

const initialForm = {
  age: 45,
  sex: 1,
  on_thyroxine: 0,
  query_on_thyroxine: 0,
  on_antithyroid_medication: 0,
  sick: 0,
  pregnant: 0,
  thyroid_surgery: 0,
  I131_treatment: 0,
  query_hypothyroid: 0,
  query_hyperthyroid: 0,
  lithium: 0,
  goitre: 0,
  tumor: 0,
  hypopituitary: 0,
  psych: 0,
  TSH_measured: 1,
  TSH: 2.5,
  T3_measured: 1,
  TT4_measured: 1,
  TT4: 110,
  T4U_measured: 1,
  T4U: 1.0,
  FTI_measured: 1,
  FTI: 110,
};

const fields = [
  ["age", "Age", "number"],
  ["sex", "Sex", "select"],
  ["TSH", "TSH", "number"],
  ["TT4", "TT4", "number"],
  ["T4U", "T4U", "number"],
  ["FTI", "FTI", "number"],
];

const medicalFields = [
  ["on_thyroxine", "On Thyroxine"],
  ["query_on_thyroxine", "Query On Thyroxine"],
  ["on_antithyroid_medication", "On Antithyroid Medication"],
  ["sick", "Sick"],
  ["pregnant", "Pregnant"],
  ["thyroid_surgery", "Thyroid Surgery"],
  ["I131_treatment", "I131 Treatment"],
  ["query_hypothyroid", "Query Hypothyroid"],
  ["query_hyperthyroid", "Query Hyperthyroid"],
  ["lithium", "Lithium"],
  ["goitre", "Goitre"],
  ["tumor", "Tumor"],
  ["hypopituitary", "Hypopituitary"],
  ["psych", "Psych"],
];

function App() {
  const [page, setPage] = useState("home");
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [error, setError] = useState("");

  const updateField = (key, value) => {
    setForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const scrollTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const goTo = (nextPage) => {
    setPage(nextPage);
    setMobileMenu(false);
    scrollTop();
  };

  const handlePredict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const payload = {};

      Object.entries(form).forEach(([key, value]) => {
        payload[key] = Number(value);
      });

      const response = await axios.post(`${API_URL}/predict`, payload, {
        timeout: 60000,
      });

      setResult(response.data);
      setPage("results");
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the prediction server. Please make sure the Render backend is online and the /predict endpoint is available."
      );
    } finally {
      setLoading(false);
    }
  };

  const prediction =
    result?.prediction === 1 ||
    result?.prediction === "1" ||
    result?.prediction_label === "Thyroid Disease Predicted";

  return (
    <div className="app-shell">
      <header className="navbar">
        <button className="brand" onClick={() => goTo("home")}>
          <span className="brand-icon">
            <HeartPulse size={22} />
          </span>

          <span>
            <strong>ThyroCare</strong>
            <small>AI DIAGNOSIS</small>
          </span>
        </button>

        <nav className={`nav-links ${mobileMenu ? "mobile-open" : ""}`}>
          <button
            className={page === "home" ? "active" : ""}
            onClick={() => goTo("home")}
          >
            Home
          </button>

          <button
            className={page === "predict" ? "active" : ""}
            onClick={() => goTo("predict")}
          >
            Prediction
          </button>

          <button
            className={page === "about" ? "active" : ""}
            onClick={() => goTo("about")}
          >
            About
          </button>

          {mobileMenu && (
            <button onClick={() => setMobileMenu(false)}>Close</button>
          )}
        </nav>

        <button
          className="mobile-menu-button"
          onClick={() => setMobileMenu((value) => !value)}
        >
          {mobileMenu ? <X size={23} /> : <Menu size={23} />}
        </button>

        <button className="nav-cta" onClick={() => goTo("predict")}>
          Start Test
          <ArrowRight size={17} />
        </button>
      </header>

      <main>
        {page === "home" && (
          <HomePage
            onStart={() => goTo("predict")}
            onAbout={() => goTo("about")}
          />
        )}

        {page === "predict" && (
          <PredictionPage
            form={form}
            updateField={updateField}
            handlePredict={handlePredict}
            loading={loading}
            error={error}
            onHome={() => goTo("home")}
          />
        )}

        {page === "results" && (
          <ResultsPage
            result={result}
            prediction={prediction}
            onNewTest={() => {
              setForm(initialForm);
              setResult(null);
              setError("");
              goTo("predict");
            }}
          />
        )}

        {page === "about" && <AboutPage onStart={() => goTo("predict")} />}
      </main>

      <footer className="footer">
        <div className="footer-brand">
          <span className="brand-icon">
            <HeartPulse size={20} />
          </span>

          <div>
            <strong>ThyroCare AI</strong>
            <p>Intelligent thyroid disease prediction.</p>
          </div>
        </div>

        <div className="footer-note">
          <ShieldCheck size={17} />
          <span>For educational and research purposes.</span>
        </div>
      </footer>
    </div>
  );
}

function HomePage({ onStart, onAbout }) {
  return (
    <section className="hero-section">
      <div className="hero-glow glow-one" />
      <div className="hero-glow glow-two" />

      <div className="hero-content">
        <div className="eyebrow">
          <Sparkles size={16} />
          AI-POWERED THYROID ANALYSIS
        </div>

        <h1>
          Understand your thyroid
          <span> with intelligent AI.</span>
        </h1>

        <p className="hero-description">
          ThyroCare combines machine learning with explainable AI to provide
          an easy-to-understand thyroid disease prediction and insights into
          the factors influencing the result.
        </p>

        <div className="hero-actions">
          <button className="primary-button" onClick={onStart}>
            Start Prediction
            <ArrowRight size={19} />
          </button>

          <button className="secondary-button" onClick={onAbout}>
            Learn About AI
            <ChevronRight size={18} />
          </button>
        </div>

        <div className="trust-row">
          <div>
            <Brain size={18} />
            <span>Machine Learning</span>
          </div>

          <div>
            <ShieldCheck size={18} />
            <span>Explainable AI</span>
          </div>

          <div>
            <Activity size={18} />
            <span>Instant Analysis</span>
          </div>
        </div>
      </div>

      <div className="hero-visual">
        <div className="medical-card main-medical-card">
          <div className="card-top">
            <span className="status-dot" />
            <span>AI ANALYSIS</span>
            <Activity size={18} />
          </div>

          <div className="heart-visual">
            <div className="pulse-ring ring-one" />
            <div className="pulse-ring ring-two" />
            <div className="heart-center">
              <HeartPulse size={48} />
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

        <div className="floating-card floating-one">
          <Brain size={20} />
          <div>
            <strong>XGBoost</strong>
            <small>Prediction model</small>
          </div>
        </div>

        <div className="floating-card floating-two">
          <Sparkles size={19} />
          <div>
            <strong>SHAP</strong>
            <small>Explainable result</small>
          </div>
        </div>
      </div>
    </section>
  );
}

function PredictionPage({
  form,
  updateField,
  handlePredict,
  loading,
  error,
  onHome,
}) {
  return (
    <section className="page-section">
      <div className="page-heading">
        <button className="back-button" onClick={onHome}>
          <Home size={16} />
          Home
        </button>

        <div className="eyebrow">
          <Stethoscope size={16} />
          THYROID PREDICTION
        </div>

        <h2>Enter patient information</h2>

        <p>
          Enter the available clinical values below. The AI model will analyze
          the information and return a prediction with explainable insights.
        </p>
      </div>

      <form className="prediction-form" onSubmit={handlePredict}>
        <div className="form-card">
          <div className="form-card-heading">
            <div className="section-icon">
              <Activity size={21} />
            </div>

            <div>
              <h3>Clinical measurements</h3>
              <p>Enter the patient's thyroid-related measurements.</p>
            </div>
          </div>

          <div className="form-grid">
            {fields.map(([key, label, type]) => (
              <label className="input-group" key={key}>
                <span>{label}</span>

                {type === "select" ? (
                  <select
                    value={form[key]}
                    onChange={(e) => updateField(key, e.target.value)}
                  >
                    <option value="0">Female / 0</option>
                    <option value="1">Male / 1</option>
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={form[key]}
                    onChange={(e) => updateField(key, e.target.value)}
                  />
                )}
              </label>
            ))}
          </div>
        </div>

        <div className="form-card">
          <div className="form-card-heading">
            <div className="section-icon">
              <HeartPulse size={21} />
            </div>

            <div>
              <h3>Clinical indicators</h3>
              <p>Select 1 if the condition or medication is present.</p>
            </div>
          </div>

          <div className="indicator-grid">
            {medicalFields.map(([key, label]) => (
              <label className="toggle-row" key={key}>
                <span>{label}</span>

                <select
                  value={form[key]}
                  onChange={(e) => updateField(key, e.target.value)}
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </label>
            ))}
          </div>
        </div>

        <div className="form-card">
          <div className="form-card-heading">
            <div className="section-icon">
              <FileText size={21} />
            </div>

            <div>
              <h3>Measurement availability</h3>
              <p>Tell the model which laboratory measurements are available.</p>
            </div>
          </div>

          <div className="indicator-grid">
            {[
              ["TSH_measured", "TSH measured"],
              ["T3_measured", "T3 measured"],
              ["TT4_measured", "TT4 measured"],
              ["T4U_measured", "T4U measured"],
              ["FTI_measured", "FTI measured"],
            ].map(([key, label]) => (
              <label className="toggle-row" key={key}>
                <span>{label}</span>

                <select
                  value={form[key]}
                  onChange={(e) => updateField(key, e.target.value)}
                >
                  <option value="1">Measured</option>
                  <option value="0">Not measured</option>
                </select>
              </label>
            ))}
          </div>
        </div>

        {error && (
          <div className="error-box">
            <CircleHelp size={20} />
            <span>{error}</span>
          </div>
        )}

        <button className="predict-button" type="submit" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="spin" size={21} />
              Analyzing...
            </>
          ) : (
            <>
              Analyze with AI
              <ArrowRight size={20} />
            </>
          )}
        </button>
      </form>
    </section>
  );
}

function ResultsPage({ result, prediction, onNewTest }) {
  if (!result) {
    return (
      <section className="page-section empty-result">
        <CircleHelp size={42} />
        <h2>No prediction available</h2>
        <p>Please complete the prediction form first.</p>
        <button className="primary-button" onClick={onNewTest}>
          Start Prediction
        </button>
      </section>
    );
  }

  const probabilities = result.probabilities || {};
  const positiveProbability =
    Number(
      probabilities["1"] ??
        probabilities[1] ??
        probabilities.positive ??
        result.probability_positive ??
        0
    ) * (Number(probabilities["1"] ?? probabilities[1]) > 1 ? 0.01 : 1);

  const negativeProbability =
    Number(
      probabilities["0"] ??
        probabilities[0] ??
        probabilities.negative ??
        result.probability_negative ??
        0
    ) * (Number(probabilities["0"] ?? probabilities[0]) > 1 ? 0.01 : 1);

  const shap = Array.isArray(result.shap)
    ? result.shap
    : Array.isArray(result.shap_values)
      ? result.shap_values
      : [];

  const counterfactuals = Array.isArray(result.counterfactuals)
    ? result.counterfactuals
    : [];

  return (
    <section className="page-section">
      <div className="page-heading">
        <div className="eyebrow">
          <Sparkles size={16} />
          AI ANALYSIS COMPLETE
        </div>

        <h2>Your prediction result</h2>

        <p>
          The machine learning model has analyzed the submitted clinical
          information.
        </p>
      </div>

      <div className={`result-banner ${prediction ? "positive" : "negative"}`}>
        <div className="result-icon">
          {prediction ? (
            <Activity size={34} />
          ) : (
            <CheckCircle2 size={34} />
          )}
        </div>

        <div>
          <span>MODEL PREDICTION</span>

          <h2>
            {prediction
              ? "Thyroid Disease Predicted"
              : "Thyroid Disease Not Predicted"}
          </h2>

          <p>
            {result.model
              ? `Model: ${result.model}`
              : "Prediction generated by the trained machine learning model."}
          </p>
        </div>
      </div>

      <div className="results-grid">
        <div className="result-card">
          <div className="result-card-title">
            <Activity size={20} />
            <span>Class probabilities</span>
          </div>

          <div className="probability">
            <div className="probability-head">
              <span>Class 0</span>
              <strong>{(negativeProbability * 100).toFixed(2)}%</strong>
            </div>

            <div className="progress">
              <div
                className="progress-bar"
                style={{
                  width: `${Math.min(100, negativeProbability * 100)}%`,
                }}
              />
            </div>
          </div>

          <div className="probability">
            <div className="probability-head">
              <span>Class 1</span>
              <strong>{(positiveProbability * 100).toFixed(2)}%</strong>
            </div>

            <div className="progress">
              <div
                className="progress-bar"
                style={{
                  width: `${Math.min(100, positiveProbability * 100)}%`,
                }}
              />
            </div>
          </div>
        </div>

        <div className="result-card">
          <div className="result-card-title">
            <Brain size={20} />
            <span>Explainable AI</span>
          </div>

          <p className="result-text">
            SHAP explains which input features contributed to the model's
            prediction. Larger absolute values generally indicate stronger
            influence on the model output.
          </p>

          {shap.length > 0 ? (
            <div className="shap-list">
              {shap.slice(0, 8).map((item, index) => {
                const name =
                  item?.feature ??
                  item?.name ??
                  item?.feature_name ??
                  `Feature ${index + 1}`;

                const value = Number(
                  item?.value ?? item?.impact ?? item?.shap ?? 0
                );

                return (
                  <div className="shap-row" key={`${name}-${index}`}>
                    <span>{name}</span>
                    <strong>{value.toFixed(4)}</strong>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty-small">
              SHAP explanation will appear here when returned by the backend.
            </div>
          )}
        </div>
      </div>

      <div className="result-card wide-card">
        <div className="result-card-title">
          <Sparkles size={20} />
          <span>Counterfactual explanation</span>
        </div>

        <p className="result-text">
          Counterfactual explanations describe how changing selected input
          values could potentially change the model's prediction.
        </p>

        {counterfactuals.length > 0 ? (
          <div className="counterfactual-list">
            {counterfactuals.slice(0, 10).map((item, index) => (
              <div className="counterfactual-item" key={index}>
                {typeof item === "object"
                  ? JSON.stringify(item)
                  : String(item)}
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-small">
            Counterfactual results will appear here when available from the
            backend.
          </div>
        )}
      </div>

      <div className="result-actions">
        <button className="primary-button" onClick={onNewTest}>
          New Prediction
          <ArrowRight size={18} />
        </button>
      </div>

      <div className="medical-disclaimer">
        <Info size={18} />
        <span>
          This application is intended for educational and research purposes.
          It is not a medical diagnosis and should not replace advice from a
          qualified healthcare professional.
        </span>
      </div>
    </section>
  );
}

function AboutPage({ onStart }) {
  return (
    <section className="page-section">
      <div className="page-heading">
        <div className="eyebrow">
          <Brain size={16} />
          ABOUT THYROCARE AI
        </div>

        <h2>Machine learning made understandable.</h2>

        <p>
          ThyroCare is designed to demonstrate how machine learning and
          explainable AI can be combined for thyroid disease prediction.
        </p>
      </div>

      <div className="about-grid">
        <div className="about-card">
          <Brain size={27} />
          <h3>Machine Learning</h3>
          <p>
            The application uses a trained machine learning model to process
            clinical features and generate a prediction.
          </p>
        </div>

        <div className="about-card">
          <Sparkles size={27} />
          <h3>Explainable AI</h3>
          <p>
            SHAP-based explanations help show which features influence the
            model output.
          </p>
        </div>

        <div className="about-card">
          <ShieldCheck size={27} />
          <h3>Transparent Results</h3>
          <p>
            The results page presents prediction probabilities and available
            explanations rather than only displaying a class label.
          </p>
        </div>

        <div className="about-card">
          <HeartPulse size={27} />
          <h3>Research Project</h3>
          <p>
            This interface is intended to support a final-year academic
            project demonstrating AI-assisted thyroid analysis.
          </p>
        </div>
      </div>

      <div className="about-cta">
        <div>
          <h3>Ready to try the prediction?</h3>
          <p>Enter the clinical values and let the model analyze them.</p>
        </div>

        <button className="primary-button" onClick={onStart}>
          Start Prediction
          <ArrowRight size={18} />
        </button>
      </div>
    </section>
  );
}

export default App;
