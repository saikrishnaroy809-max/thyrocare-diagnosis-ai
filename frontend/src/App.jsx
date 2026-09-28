import { useState } from "react";
import axios from "axios";
import {
  Activity,
  ArrowRight,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  FileText,
  FlaskConical,
  Gauge,
  HeartPulse,
  Info,
  LayoutDashboard,
  LockKeyhole,
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
  T4U: 1,
  FTI_measured: 1,
  FTI: 110,
};

const fieldLabels = {
  age: "Age",
  sex: "Sex",
  on_thyroxine: "On Thyroxine",
  query_on_thyroxine: "Query on Thyroxine",
  on_antithyroid_medication: "On Antithyroid Medication",
  sick: "Sick",
  pregnant: "Pregnant",
  thyroid_surgery: "Thyroid Surgery",
  I131_treatment: "I131 Treatment",
  query_hypothyroid: "Query Hypothyroid",
  query_hyperthyroid: "Query Hyperthyroid",
  lithium: "Lithium",
  goitre: "Goitre",
  tumor: "Tumor",
  hypopituitary: "Hypopituitary",
  psych: "Psych",
  TSH_measured: "TSH Measured",
  TSH: "TSH",
  T3_measured: "T3 Measured",
  TT4_measured: "TT4 Measured",
  TT4: "TT4",
  T4U_measured: "T4U Measured",
  T4U: "T4U",
  FTI_measured: "FTI Measured",
  FTI: "FTI",
};

const binaryFields = [
  "on_thyroxine",
  "query_on_thyroxine",
  "on_antithyroid_medication",
  "sick",
  "pregnant",
  "thyroid_surgery",
  "I131_treatment",
  "query_hypothyroid",
  "query_hyperthyroid",
  "lithium",
  "goitre",
  "tumor",
  "hypopituitary",
  "psych",
];

const measuredFields = [
  "TSH_measured",
  "T3_measured",
  "TT4_measured",
  "T4U_measured",
  "FTI_measured",
];

function App() {
  const [page, setPage] = useState("home");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const navigate = (nextPage) => {
    setPage(nextPage);
    setMobileMenu(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const updateField = (key, value) => {
    setForm((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const predict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const payload = {
        ...form,
        age: Number(form.age),
        sex: Number(form.sex),
        TSH: Number(form.TSH),
        TT4: Number(form.TT4),
        T4U: Number(form.T4U),
        FTI: Number(form.FTI),
      };

      Object.keys(payload).forEach((key) => {
        if (key !== "age" && key !== "TSH" && key !== "TT4" && key !== "T4U" && key !== "FTI") {
          payload[key] = Number(payload[key]);
        }
      });

      const response = await axios.post(`${API_URL}/predict`, payload);

      setResult(response.data);
      setPage("results");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to connect to the prediction server. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetPrediction = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
    setPage("predict");
  };

  return (
    <div className="app-shell">
      <header className="navbar">
        <div className="nav-inner">
          <button className="brand" onClick={() => navigate("home")}>
            <span className="brand-icon">
              <HeartPulse size={22} />
            </span>
            <span>
              <strong>ThyroCare</strong>
              <small>AI • Explainable • Intelligent</small>
            </span>
          </button>

          <nav className={`nav-links ${mobileMenu ? "open" : ""}`}>
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

            <button
              className="admin-nav"
              onClick={() => navigate("admin")}
            >
              <LockKeyhole size={15} />
              Admin
            </button>
          </nav>

          <button
            className="mobile-menu-button"
            onClick={() => setMobileMenu((value) => !value)}
          >
            {mobileMenu ? <X /> : <Menu />}
          </button>
        </div>
      </header>

      <main>
        {page === "home" && (
          <HomePage navigate={navigate} />
        )}

        {page === "predict" && (
          <PredictionPage
            form={form}
            updateField={updateField}
            predict={predict}
            loading={loading}
            error={error}
          />
        )}

        {page === "results" && (
          <ResultsPage
            result={result}
            navigate={navigate}
            resetPrediction={resetPrediction}
          />
        )}

        {page === "about" && <AboutPage />}

        {page === "admin" && <AdminPage />}
      </main>

      <footer className="footer">
        <div>
          <strong>ThyroCare</strong>
          <span> AI-powered thyroid disease analysis</span>
        </div>
        <span>Built with Machine Learning & Explainable AI</span>
      </footer>
    </div>
  );
}

function HomePage({ navigate }) {
  return (
    <section className="hero-page">
      <div className="hero-glow glow-one" />
      <div className="hero-glow glow-two" />

      <div className="hero-content">
        <div className="eyebrow">
          <Sparkles size={16} />
          Explainable AI for Thyroid Analysis
        </div>

        <h1>
          Smarter thyroid analysis.
          <span>Clearer explanations.</span>
        </h1>

        <p className="hero-description">
          ThyroCare combines machine learning with explainable AI to provide
          an understandable analysis of thyroid-related patient data.
        </p>

        <div className="hero-actions">
          <button className="primary-button" onClick={() => navigate("predict")}>
            Start Prediction
            <ArrowRight size={18} />
          </button>

          <button className="secondary-button" onClick={() => navigate("about")}>
            Explore ThyroCare
            <ChevronRight size={18} />
          </button>
        </div>

        <div className="trust-row">
          <div>
            <ShieldCheck size={19} />
            <span>Explainable results</span>
          </div>
          <div>
            <Brain size={19} />
            <span>XGBoost AI</span>
          </div>
          <div>
            <Activity size={19} />
            <span>SHAP analysis</span>
          </div>
        </div>
      </div>

      <div className="hero-card-area">
        <div className="ai-card main-ai-card">
          <div className="card-top">
            <div className="mini-icon">
              <Brain size={21} />
            </div>
            <span className="live-pill">
              <span />
              AI Ready
            </span>
          </div>

          <div className="scan-circle">
            <HeartPulse size={55} />
            <div className="scan-ring ring-one" />
            <div className="scan-ring ring-two" />
          </div>

          <h3>Thyroid AI Engine</h3>
          <p>Prediction + explainability in one workflow</p>

          <div className="confidence-card">
            <div>
              <span>Model</span>
              <strong>XGBoost</strong>
            </div>
            <div>
              <span>Features</span>
              <strong>25</strong>
            </div>
          </div>
        </div>

        <div className="floating-card floating-one">
          <CheckCircle2 size={18} />
          <span>Prediction ready</span>
        </div>

        <div className="floating-card floating-two">
          <Sparkles size={18} />
          <span>SHAP enabled</span>
        </div>
      </div>
    </section>
  );
}

function PredictionPage({
  form,
  updateField,
  predict,
  loading,
  error,
}) {
  return (
    <section className="page-container">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <FlaskConical size={16} />
            AI Prediction
          </div>

          <h2>Enter patient information</h2>

          <p>
            Provide the available thyroid-related measurements and clinical
            indicators. The model will analyze the 25 features.
          </p>
        </div>

        <div className="model-badge">
          <Brain size={18} />
          XGBoost
        </div>
      </div>

      <form className="prediction-form" onSubmit={predict}>
        <section className="form-section">
          <div className="section-title">
            <Stethoscope size={20} />
            <div>
              <h3>Basic information</h3>
              <p>Patient demographic information</p>
            </div>
          </div>

          <div className="field-grid three">
            <NumberField
              label="Age"
              value={form.age}
              onChange={(value) => updateField("age", value)}
              min="1"
              max="120"
              step="1"
            />

            <SelectField
              label="Sex"
              value={form.sex}
              onChange={(value) => updateField("sex", value)}
              options={[
                { value: 0, label: "Female / 0" },
                { value: 1, label: "Male / 1" },
              ]}
            />
          </div>
        </section>

        <section className="form-section">
          <div className="section-title">
            <Activity size={20} />
            <div>
              <h3>Clinical indicators</h3>
              <p>Select 1 when the condition or indicator is present</p>
            </div>
          </div>

          <div className="field-grid three">
            {binaryFields.map((field) => (
              <SelectField
                key={field}
                label={fieldLabels[field]}
                value={form[field]}
                onChange={(value) => updateField(field, value)}
                options={[
                  { value: 0, label: "No / 0" },
                  { value: 1, label: "Yes / 1" },
                ]}
              />
            ))}
          </div>
        </section>

        <section className="form-section">
          <div className="section-title">
            <Gauge size={20} />
            <div>
              <h3>Laboratory measurements</h3>
              <p>Enter measured values where available</p>
            </div>
          </div>

          <div className="field-grid three">
            {measuredFields.map((field) => (
              <SelectField
                key={field}
                label={fieldLabels[field]}
                value={form[field]}
                onChange={(value) => updateField(field, value)}
                options={[
                  { value: 0, label: "Not measured / 0" },
                  { value: 1, label: "Measured / 1" },
                ]}
              />
            ))}

            <NumberField
              label="TSH"
              value={form.TSH}
              onChange={(value) => updateField("TSH", value)}
              step="0.01"
            />

            <NumberField
              label="TT4"
              value={form.TT4}
              onChange={(value) => updateField("TT4", value)}
              step="0.01"
            />

            <NumberField
              label="T4U"
              value={form.T4U}
              onChange={(value) => updateField("T4U", value)}
              step="0.01"
            />

            <NumberField
              label="FTI"
              value={form.FTI}
              onChange={(value) => updateField("FTI", value)}
              step="0.01"
            />
          </div>
        </section>

        {error && (
          <div className="error-box">
            <Info size={20} />
            <div>
              <strong>Prediction failed</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        <div className="form-actions">
          <button className="primary-button large" type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                Analyze with AI
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>
      </form>
    </section>
  );
}

function ResultsPage({ result, navigate, resetPrediction }) {
  if (!result) {
    return (
      <section className="empty-state">
        <FileText size={45} />
        <h2>No prediction available</h2>
        <p>Run a prediction first to see the analysis.</p>
        <button className="primary-button" onClick={() => navigate("predict")}>
          Go to Prediction
        </button>
      </section>
    );
  }

  const class0 = Number(result.probabilities?.class_0 || 0);
  const class1 = Number(result.probabilities?.class_1 || 0);
  const predictedDisease = result.prediction === 1;

  const shapFeatures = [...(result.shap?.features || [])]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 8);

  return (
    <section className="page-container results-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <Sparkles size={16} />
            AI Analysis Complete
          </div>
          <h2>Prediction results</h2>
          <p>
            The model has analyzed the submitted 25-feature patient profile.
          </p>
        </div>

        <div className="model-badge">
          <Brain size={18} />
          {result.model?.name || "XGBoost"}
        </div>
      </div>

      <div className={`result-banner ${predictedDisease ? "positive" : "negative"}`}>
        <div className="result-icon">
          {predictedDisease ? <Activity size={30} /> : <CheckCircle2 size={30} />}
        </div>

        <div>
          <span>Model prediction</span>
          <h3>{result.prediction_label}</h3>
          <p>
            This is an AI model output and should not be treated as a medical
            diagnosis.
          </p>
        </div>
      </div>

      <div className="results-grid">
        <div className="result-panel probability-panel">
          <div className="panel-heading">
            <div>
              <h3>Class probabilities</h3>
              <p>Model confidence distribution</p>
            </div>
            <Gauge size={22} />
          </div>

          <ProbabilityBar
            label="Class 0 — Not Predicted"
            value={class0}
          />

          <ProbabilityBar
            label="Class 1 — Predicted"
            value={class1}
          />
        </div>

        <div className="result-panel">
          <div className="panel-heading">
            <div>
              <h3>Model information</h3>
              <p>Current prediction engine</p>
            </div>
            <Database size={22} />
          </div>

          <div className="info-list">
            <InfoRow label="Algorithm" value={result.model?.name || "XGBoost"} />
            <InfoRow label="Version" value={result.model?.version || "Tuned"} />
            <InfoRow
              label="Features analyzed"
              value={result.model?.feature_count || 25}
            />
            <InfoRow
              label="SHAP"
              value={result.shap?.available ? "Available" : "Unavailable"}
            />
          </div>
        </div>
      </div>

      <div className="result-panel shap-panel">
        <div className="panel-heading">
          <div>
            <h3>Explainable AI — SHAP</h3>
            <p>Features with the largest contribution to this prediction</p>
          </div>
          <Sparkles size={22} />
        </div>

        {shapFeatures.length > 0 ? (
          <div className="shap-list">
            {shapFeatures.map((item, index) => (
              <div className="shap-row" key={`${item.feature}-${index}`}>
                <div className="shap-name">
                  <strong>{item.feature}</strong>
                  <span>Value: {String(item.value)}</span>
                </div>

                <div className="shap-impact">
                  <div className="impact-track">
                    <div
                      className={`impact-fill ${
                        item.impact >= 0 ? "impact-positive" : "impact-negative"
                      }`}
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(5, Math.abs(item.impact) * 18)
                        )}%`,
                      }}
                    />
                  </div>

                  <span>
                    {item.impact > 0 ? "+" : ""}
                    {Number(item.impact).toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-inline">
            SHAP explanation is not available for this prediction.
          </div>
        )}
      </div>

      <div className="result-panel">
        <div className="panel-heading">
          <div>
            <h3>Counterfactual explanation</h3>
            <p>What-if analysis can be added to the model workflow</p>
          </div>
          <Brain size={22} />
        </div>

        <div className="counterfactual-card">
          <Sparkles size={24} />
          <div>
            <strong>Explainable AI workflow</strong>
            <p>
              The current backend provides SHAP explanations. Counterfactual
              generation can be connected here as the next XAI module.
            </p>
          </div>
        </div>
      </div>

      <div className="result-actions">
        <button className="secondary-button" onClick={resetPrediction}>
          New Prediction
        </button>

        <button className="primary-button" onClick={() => window.print()}>
          <FileText size={18} />
          Print / Save Report
        </button>
      </div>
    </section>
  );
}

function AboutPage() {
  return (
    <section className="page-container">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <Info size={16} />
            About the project
          </div>

          <h2>ThyroCare</h2>

          <p>
            A final-year machine learning project focused on thyroid disease
            analysis and explainable artificial intelligence.
          </p>
        </div>
      </div>

      <div className="feature-cards">
        <FeatureCard
          icon={<Brain />}
          title="Machine Learning"
          text="An XGBoost model analyzes 25 thyroid-related features."
        />

        <FeatureCard
          icon={<Sparkles />}
          title="Explainable AI"
          text="SHAP helps show which input features contributed to a prediction."
        />

        <FeatureCard
          icon={<ShieldCheck />}
          title="Transparent Results"
          text="Prediction probabilities and model information are presented clearly."
        />

        <FeatureCard
          icon={<LayoutDashboard />}
          title="Admin Analytics"
          text="The project architecture supports an administrative dashboard for dataset and model analysis."
        />
      </div>
    </section>
  );
}

function AdminPage() {
  return (
    <section className="page-container">
      <div className="admin-login-card">
        <div className="admin-logo">
          <LockKeyhole size={30} />
        </div>

        <div className="eyebrow">
          <ShieldCheck size={15} />
          Administrator Access
        </div>

        <h2>Admin Dashboard</h2>

        <p>
          The administration module will provide dataset management,
          preprocessing, algorithm comparison, model metrics, and prediction
          history.
        </p>

        <div className="admin-preview-grid">
          <div>
            <Database size={20} />
            <span>Dataset Management</span>
          </div>

          <div>
            <Gauge size={20} />
            <span>Model Metrics</span>
          </div>

          <div>
            <Activity size={20} />
            <span>Prediction History</span>
          </div>

          <div>
            <LayoutDashboard size={20} />
            <span>Analytics Dashboard</span>
          </div>
        </div>

        <div className="admin-note">
          <Info size={18} />
          <span>
            Admin authentication and management APIs will be connected in the
            next development stage.
          </span>
        </div>
      </div>
    </section>
  );
}

function NumberField({ label, value, onChange, min, max, step = "1" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {options.map((option) => (
          <option key={`${option.value}-${option.label}`} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ProbabilityBar({ label, value }) {
  return (
    <div className="probability-item">
      <div className="probability-label">
        <span>{label}</span>
        <strong>{(value * 100).toFixed(2)}%</strong>
      </div>

      <div className="probability-track">
        <div
          className="probability-fill"
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FeatureCard({ icon, title, text }) {
  return (
    <div className="feature-card">
      <div className="feature-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

export default App;
