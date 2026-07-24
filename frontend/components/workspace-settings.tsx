"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useAuth } from "@/components/auth-provider";
import {
  createWorkspaceInvitation,
  getBillingAccount,
  listAuditEvents,
  listBillingEntries,
  listWorkspaceInvitations,
  listWorkspaceMembers,
  removeWorkspaceMember,
  requestEmailVerification,
  revokeWorkspaceInvitation,
  updateWorkspaceMember,
} from "@/lib/api";
import type {
  AuditEvent,
  BillingAccount,
  BillingEntry,
  WorkspaceInvitation,
  WorkspaceMember,
  WorkspaceRole,
} from "@/lib/types";

const ROLES: WorkspaceRole[] = ["owner", "admin", "editor", "viewer"];

function number(value: string): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(2) : value;
}

export function WorkspaceSettings() {
  const params = useParams<{ workspaceId: string }>();
  const workspaceId = params.workspaceId;
  const { session, loading, refresh } = useAuth();
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [invitations, setInvitations] = useState<WorkspaceInvitation[]>([]);
  const [billing, setBilling] = useState<BillingAccount | null>(null);
  const [entries, setEntries] = useState<BillingEntry[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<WorkspaceRole>("editor");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const workspace = useMemo(
    () => session?.workspaces.find((item) => item.id === workspaceId),
    [session, workspaceId],
  );
  const canManage = workspace?.role === "owner" || workspace?.role === "admin";

  const load = useCallback(async () => {
    if (!workspace) return;
    try {
      const [nextBilling, nextEntries] = await Promise.all([
        getBillingAccount(workspaceId),
        listBillingEntries(workspaceId),
      ]);
      setBilling(nextBilling);
      setEntries(nextEntries);
      if (canManage) {
        const [nextMembers, nextInvitations, nextEvents] = await Promise.all([
          listWorkspaceMembers(workspaceId),
          listWorkspaceInvitations(workspaceId),
          listAuditEvents(workspaceId),
        ]);
        setMembers(nextMembers);
        setInvitations(nextInvitations);
        setEvents(nextEvents);
      }
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load workspace operations.");
    }
  }, [canManage, workspace, workspaceId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function invite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await createWorkspaceInvitation(workspaceId, email, role);
      setEmail("");
      setMessage("Invitation queued for delivery.");
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invitation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function changeRole(member: WorkspaceMember, nextRole: WorkspaceRole) {
    setBusy(true);
    try {
      await updateWorkspaceMember(workspaceId, member.id, nextRole);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Role update failed.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(member: WorkspaceMember) {
    if (!window.confirm(`Remove ${member.email} from this workspace?`)) return;
    setBusy(true);
    try {
      await removeWorkspaceMember(workspaceId, member.id);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Member removal failed.");
    } finally {
      setBusy(false);
    }
  }

  async function verifyEmail() {
    setBusy(true);
    try {
      const response = await requestEmailVerification();
      setMessage(response.message);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Verification request failed.");
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <main className="shell"><div className="panel loading">Loading workspace…</div></main>;
  if (!session || !workspace) return <main className="shell"><div className="panel loading">Workspace unavailable.</div></main>;

  return (
    <main className="shell operations-shell">
      <header className="workspace-bar">
        <div>
          <Link className="back" href="/">← Production library</Link>
          <div className="eyebrow">WORKSPACE OPERATIONS</div>
          <h1>{workspace.name}</h1>
          <p>Role: {workspace.role} · account security, team access, credits, and audit history.</p>
        </div>
      </header>

      {!session.user.email_verified && (
        <section className="alert progress verification-banner">
          <div><strong>Verify {session.user.email}</strong><p>Verification protects recovery and invitation workflows.</p></div>
          <button className="primary" type="button" disabled={busy} onClick={verifyEmail}>Send verification</button>
        </section>
      )}
      {message && <div className="alert success">{message}</div>}
      {error && <div className="alert error">{error}</div>}

      <div className="operations-grid">
        <section className="panel operation-card">
          <div className="eyebrow">CREDITS</div>
          <h2>{billing ? number(billing.available_credits) : "—"} available</h2>
          <p>{billing ? `${number(billing.balance_credits)} total · ${number(billing.reserved_credits)} reserved` : "Loading ledger…"}</p>
          <div className="ledger-list">
            {entries.slice(0, 8).map((entry) => (
              <article key={entry.id}><span>{entry.kind}</span><strong>{number(entry.amount_credits)}</strong><small>{entry.description}</small></article>
            ))}
          </div>
        </section>

        {canManage && (
          <section className="panel operation-card">
            <div className="eyebrow">INVITE TEAM</div>
            <h2>Workspace access</h2>
            <form className="inline-form" onSubmit={invite}>
              <input type="email" placeholder="editor@example.com" value={email} onChange={(event) => setEmail(event.target.value)} required />
              <select value={role} onChange={(event) => setRole(event.target.value as WorkspaceRole)}>
                {ROLES.filter((item) => item !== "owner").map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
              <button className="primary" disabled={busy}>Invite</button>
            </form>
            <div className="ledger-list">
              {invitations.filter((item) => !item.accepted_at && !item.revoked_at).map((item) => (
                <article key={item.id}><span>{item.email}</span><strong>{item.role}</strong><button className="text-button" type="button" onClick={() => revokeWorkspaceInvitation(workspaceId, item.id).then(load)}>Revoke</button></article>
              ))}
            </div>
          </section>
        )}
      </div>

      {canManage && (
        <section className="panel operation-card full-width">
          <div className="eyebrow">MEMBERS</div>
          <h2>{members.length} workspace member{members.length === 1 ? "" : "s"}</h2>
          <div className="member-table">
            {members.map((member) => (
              <article key={member.id}>
                <div><strong>{member.display_name}</strong><small>{member.email}</small></div>
                <select disabled={busy || member.user_id === session.user.id && member.role === "owner"} value={member.role} onChange={(event) => void changeRole(member, event.target.value as WorkspaceRole)}>
                  {ROLES.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <button className="secondary" type="button" disabled={busy || member.user_id === session.user.id} onClick={() => void remove(member)}>Remove</button>
              </article>
            ))}
          </div>
        </section>
      )}

      {canManage && (
        <section className="panel operation-card full-width">
          <div className="eyebrow">AUDIT TRAIL</div>
          <h2>Recent mutations</h2>
          <div className="audit-list">
            {events.slice(0, 50).map((event) => (
              <article key={event.id}><time>{new Date(event.created_at).toLocaleString()}</time><strong>{event.action}</strong><span>{event.resource_type}{event.resource_id ? ` · ${event.resource_id.slice(0, 8)}` : ""}</span><small>{event.request_id ?? "no request id"}</small></article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
