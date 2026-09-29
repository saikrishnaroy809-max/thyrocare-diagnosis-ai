import React, { useState } from "react";

export default function AdminLogin({
  onLogin,
  onBack,
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  /* =========================================================
     SUBMIT LOGIN
     ========================================================= */

  const handleSubmit = (event) => {
    event.preventDefault();

    setError("");

    const cleanUsername = username.trim();

    if (!cleanUsername || !password) {
      setError(
        "Please enter both username and password."
      );
      return;
    }

    /*
      Authentication is handled by App.jsx.

      This component only collects the credentials
      and sends them to the parent component.
    */

    const success = onLogin?.(
      cleanUsername,
      password
    );

    if (success === false) {
      setError(
        "Invalid admin username or password."
      );
    }
  };

  /* =========================================================
     CLEAR ERROR WHEN USER TYPES
     ========================================================= */

  const handleUsernameChange = (event) => {
    setUsername(event.target.value);

    if (error) {
      setError("");
    }
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);

    if (error) {
      setError("");
    }
  };

  /* =========================================================
     UI
     ========================================================= */

  return (
    <main className="admin-login-page">

      <div className="admin-login-card">

        {/* ===================================================
            HEADER
            =================================================== */}

        <div className="admin-login-header">

          <div
            className="admin-login-icon"
            aria-hidden="true"
          >
            🔐
          </div>

          <span className="admin-panel-badge">
            THYROCARE AI
          </span>

          <h1>
            Admin Login
          </h1>

          <p>
            Sign in to access the ThyroCare AI
            administration dashboard.
          </p>

        </div>


        {/* ===================================================
            LOGIN FORM
            =================================================== */}

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
              name="username"
              placeholder="Enter admin username"
              value={username}
              onChange={handleUsernameChange}
              autoComplete="username"
              autoCapitalize="none"
              spellCheck="false"
            />

          </div>


          {/* PASSWORD */}

          <div className="admin-field">

            <label htmlFor="admin-password">
              Password
            </label>

            <div className="admin-password-wrapper">

              <input
                id="admin-password"
                type={
                  showPassword
                    ? "text"
                    : "password"
                }
                name="password"
                placeholder="Enter admin password"
                value={password}
                onChange={handlePasswordChange}
                autoComplete="current-password"
              />

              <button
                type="button"
                className="admin-password-toggle"
                onClick={() =>
                  setShowPassword(
                    (previous) => !previous
                  )
                }
                aria-label={
                  showPassword
                    ? "Hide password"
                    : "Show password"
                }
              >
                {showPassword
                  ? "HIDE"
                  : "SHOW"}
              </button>

            </div>

          </div>


          {/* ERROR */}

          {error && (
            <div
              className="error-message"
              role="alert"
            >
              {error}
            </div>
          )}


          {/* LOGIN */}

          <button
            type="submit"
            className="admin-login-button"
          >
            Sign In to Dashboard
          </button>

        </form>


        {/* ===================================================
            DEMO INFORMATION
            =================================================== */}

        <div className="admin-demo-note">

          <strong>
            Development Demo
          </strong>

          <div className="admin-demo-row">
            <span>Username</span>
            <strong>admin</strong>
          </div>

          <div className="admin-demo-row">
            <span>Password</span>
            <strong>thyrocare123</strong>
          </div>

          <p>
            This login is frontend/demo
            authentication. Hardcoded credentials
            should not be used for production
            security.
          </p>

        </div>


        {/* ===================================================
            BACK
            =================================================== */}

        <button
          type="button"
          className="secondary-button admin-back-button"
          onClick={onBack}
        >
          ← Back to Website
        </button>

      </div>

    </main>
  );
        }
