import React, { useState } from "react";

export default function AdminLogin({
  onLogin,
  onBack,
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    setError("");

    if (!username.trim() || !password.trim()) {
      setError("Please enter both username and password.");
      return;
    }

    /*
      App.jsx handles the actual demo credential
      verification through the onLogin callback.
    */
    const success = onLogin?.(username.trim(), password);

    if (success === false) {
      setError("Invalid admin username or password.");
    }
  };

  return (
    <main className="admin-login-page">

      <div className="admin-login-card">

        {/* =====================================================
            HEADER
            ===================================================== */}

        <div className="admin-login-header">

          <div className="admin-login-icon">
            🔐
          </div>

          <h1>
            Admin Login
          </h1>

          <p>
            Sign in to access the ThyroCare AI
            administration dashboard.
          </p>

        </div>

        {/* =====================================================
            LOGIN FORM
            ===================================================== */}

        <form
          className="admin-login-form"
          onSubmit={handleSubmit}
        >

          {/* USERNAME */}

          <div className="admin-field">

            <label htmlFor="admin-username">
              Username
            </label>

            <input
              id="admin-username"
              type="text"
              placeholder="Enter admin username"
              value={username}
              onChange={(e) => {
                setUsername(e.target.value);
                setError("");
              }}
              autoComplete="username"
            />

          </div>

          {/* PASSWORD */}

          <div className="admin-field">

            <label htmlFor="admin-password">
              Password
            </label>

            <div
              style={{
                position: "relative",
              }}
            >

              <input
                id="admin-password"
                type={
                  showPassword
                    ? "text"
                    : "password"
                }
                placeholder="Enter admin password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError("");
                }}
                autoComplete="current-password"
                style={{
                  paddingRight: "75px",
                }}
              />

              <button
                type="button"
                onClick={() =>
                  setShowPassword(
                    (previous) => !previous
                  )
                }
                style={{
                  position: "absolute",
                  right: "8px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  border: "0",
                  background: "transparent",
                  color: "#94a3b8",
                  cursor: "pointer",
                  fontSize: "0.75rem",
                  fontWeight: "700",
                  padding: "7px",
                }}
              >
                {showPassword
                  ? "HIDE"
                  : "SHOW"}
              </button>

            </div>

          </div>

          {/* ERROR */}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {/* LOGIN BUTTON */}

          <button
            type="submit"
            className="admin-login-button"
          >
            Sign In to Dashboard
          </button>

        </form>

        {/* =====================================================
            DEMO INFORMATION
            ===================================================== */}

        <div className="admin-demo-note">

          <strong>
            Development Demo
          </strong>

          <br />

          Username:
          <strong> admin</strong>

          <br />

          Password:
          <strong> thyrocare123</strong>

          <br />
          <br />

          This is frontend/demo authentication.
          Do not use hardcoded credentials for
          production security.

        </div>

        {/* =====================================================
            BACK BUTTON
            ===================================================== */}

        <button
          type="button"
          className="secondary-button"
          onClick={onBack}
          style={{
            width: "100%",
            marginTop: "14px",
          }}
        >
          ← Back to Website
        </button>

      </div>

    </main>
  );
}
