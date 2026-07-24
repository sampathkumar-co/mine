"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { AuthPanel } from "@/components/auth-panel";
import { useAuth } from "@/components/auth-provider";
import {
  acceptWorkspaceInvitation,
  confirmEmailVerification,
  confirmPasswordReset,
  requestPasswordReset,
} from "@/lib/api";

export function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await requestPasswordReset(email);
      setMessage(response.message);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Password reset request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell auth-shell compact-auth">
      <section className="auth-copy"><div className="eyebrow">ACCOUNT RECOVERY</div><h1>Reset access without losing the edit.</h1><p>A short-lived reset link will be delivered through the configured email provider.</p></section>
      <form className="panel auth-panel" onSubmit={submit}>
        <label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
        {message && <div className="alert success">{message}</div>}
        {error && <div className="alert error">{error}</div>}
        <button className="primary action" disabled={busy}>{busy ? "Queuing…" : "Send reset link"}</button>
        <Link className="text-link" href="/">Back to sign in</Link>
      </form>
    </main>
  );
}

export function ResetPassword() {
  const params = useSearchParams();
  const token = params.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await confirmPasswordReset(token, password);
      setMessage(response.message);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Password reset failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell auth-shell compact-auth">
      <section className="auth-copy"><div className="eyebrow">NEW PASSWORD</div><h1>Rotate the key. Keep the production.</h1><p>Successful reset revokes every existing session for the account.</p></section>
      <form className="panel auth-panel" onSubmit={submit}>
        {!token && <div className="alert error">This reset link has no token.</div>}
        <label>New password<input type="password" minLength={10} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
        {message && <div className="alert success">{message} <Link href="/">Sign in</Link></div>}
        {error && <div className="alert error">{error}</div>}
        <button className="primary action" disabled={busy || !token}>{busy ? "Changing…" : "Change password"}</button>
      </form>
    </main>
  );
}

export function VerifyEmail() {
  const params = useSearchParams();
  const token = params.get("token") ?? "";
  const [message, setMessage] = useState("Verifying email…");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) {
      setMessage("");
      setError("This verification link has no token.");
      return;
    }
    confirmEmailVerification(token)
      .then((response) => setMessage(response.message))
      .catch((caught) => {
        setMessage("");
        setError(caught instanceof Error ? caught.message : "Email verification failed.");
      });
  }, [token]);

  return (
    <main className="shell auth-shell compact-auth">
      <section className="auth-copy"><div className="eyebrow">EMAIL VERIFICATION</div><h1>Trust the address behind the workspace.</h1></section>
      <section className="panel auth-panel">
        {message && <div className="alert success">{message}</div>}
        {error && <div className="alert error">{error}</div>}
        <Link className="primary button-link action" href="/">Open Director OS</Link>
      </section>
    </main>
  );
}

export function AcceptInvitation() {
  const params = useSearchParams();
  const token = params.get("token") ?? "";
  const { session, loading, refresh } = useAuth();
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function accept() {
    setBusy(true);
    setError("");
    try {
      await acceptWorkspaceInvitation(token);
      await refresh();
      setMessage("Invitation accepted. The workspace is now available in your library.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invitation could not be accepted.");
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <main className="shell"><div className="panel loading">Opening invitation…</div></main>;
  if (!session) return <><div className="invitation-note">Sign in with the invited email, then return to this link.</div><AuthPanel /></>;

  return (
    <main className="shell auth-shell compact-auth">
      <section className="auth-copy"><div className="eyebrow">WORKSPACE INVITATION</div><h1>Join the production room.</h1><p>Signed in as {session.user.email}.</p></section>
      <section className="panel auth-panel">
        {!token && <div className="alert error">This invitation link has no token.</div>}
        {message && <div className="alert success">{message}</div>}
        {error && <div className="alert error">{error}</div>}
        {!message && <button className="primary action" disabled={busy || !token} onClick={() => void accept()}>{busy ? "Joining…" : "Accept invitation"}</button>}
        <Link className="text-link" href="/">Open production library</Link>
      </section>
    </main>
  );
}
