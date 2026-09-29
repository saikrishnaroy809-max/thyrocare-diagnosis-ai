import React, { useState } from "react";

export default function AdminLogin({ onLogin, onBack }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (!username || !password) {
      setError("Please enter username and password.");
      return;
    }

    // Temporary login for testing.
    // Secure backend authentication will be added later.
    if (username === "admin" && password === "thyrocare") {
      onLogin();
    } else {
      setError("Invalid admin credentials.");
    }
  };

  return (
    <div className="admin-login-page">

      <div className="admin-login-card">

        <div className="admin-login-icon">
          🔐
        </div>

        <div className="admin-login-header">
          <span>THYROCARE AI</span>

          <h1>Admin Login</h1>

          <p>
            Secure access to the ThyroCare AI administration panel.
          </p>
        </div>

        <form onSubmit={handleSubmit}>

          <div className="admin-field">
            <label>Username</label>

            <input
              type="text"
              placeholder="Enter admin username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
            />
          </div>

          <div className="admin-field">
            <label>Password</label>

            <input
              type="password"
              placeholder="Enter admin password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>

          {error && (
            <div className="admin-login-error">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="admin-login-button"
          >
            Login to Dashboard →
          </button>

        </form>

        <button
          type="button"
          className="admin-back-button"
          onClick={onBack}
        >
          ← Back to ThyroCare
        </button>

        <div className="admin-login-footer">
          ThyroCare AI • Administration
        </div>

      </div>

    </div>
  );
}
