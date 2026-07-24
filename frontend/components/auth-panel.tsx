"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { useAuth } from "@/components/auth-provider";

export function AuthPanel() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [workspaceName, setWorkspaceName] = useState("My Studio");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (mode === "login") await login(email, password);
      else await register({ email, password, displayName, workspaceName });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell auth-shell">
      <section className="auth-copy">
        <div className="eyebrow">DIRECTOR OS / PRIVATE PRODUCTION</div>
        <h1>Your footage. Your workspace. Your director.</h1>
        <p>
          Sign in to keep productions, pickup missions, revisions, delivery, credits, and audit
          history inside one protected workspace.
        </p>
      </section>
      <form className="panel auth-panel" onSubmit={submit}>
        <div className="choice-row auth-choice">
          <button type="button" className={`choice ${mode === "login" ? "active" : ""}`} onClick={() => setMode("login")}>Sign in</button>
          <button type="button" className={`choice ${mode === "register" ? "active" : ""}`} onClick={() => setMode("register")}>Create account</button>
        </div>
        {mode === "register" && (
          <>
            <label>Display name<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} minLength={2} required /></label>
            <label>Workspace name<input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} minLength={2} required /></label>
          </>
        )}
        <label>Email<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
        <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(event) => setPassword(event.target.value)} minLength={mode === "register" ? 10 : 1} required /></label>
        {mode === "login" && <Link className="text-link" href="/forgot-password">Forgot password?</Link>}
        {error && <div className="alert error">{error}</div>}
        <button className="primary action" disabled={busy} type="submit">
          {busy ? "Opening workspace…" : mode === "login" ? "Sign in" : "Create workspace"}
        </button>
      </form>
    </main>
  );
}
