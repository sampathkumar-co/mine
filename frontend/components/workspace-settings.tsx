"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useAuth } from "@/components/auth-provider";
import { PrivacyPanel } from "@/components/privacy-panel";
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
import {
  createSubscriptionCheckout,
  createSubscriptionPortal,
  getSubscriptionOverview,
  listBillingPlans,
} from "@/lib/subscriptions";
import type { BillingPlan, SubscriptionOverview } from "@/lib/subscriptions";
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

function planDetail(plan: BillingPlan): string {
  return `${plan.max_source_clips} clips · ${plan.max_target_duration_seconds}s · ${plan.max_members} seats · Tier ${plan.max_tier}`;
}

export function WorkspaceSettings() {
  const params = useParams<{ workspaceId: string }>();
  const workspaceId = params.workspaceId;
  const { session, loading, refresh } = useAuth();
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [invitations, setInvitations] = useState<WorkspaceInvitation[]>([]);
  const [billing, setBilling] = useState<BillingAccount | null>(null);
  const [subscription, setSubscription] = useState<SubscriptionOverview | null>(null);
  const [plans, setPlans] = useState<BillingPlan[]>([]);
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
  const canManageBilling = workspace?.role === "owner";

  const load = useCallback(async () => {
    if (!workspace) return;
    try {
      const [nextBilling, nextEntries, nextSubscription, nextPlans] = await Promise.all([
        getBillingAccount(workspaceId),
        listBillingEntries(workspaceId),
        getSubscriptionOverview(workspaceId),
        listBillingPlans(),
      ]);
      setBilling(nextBilling);
      setEntries(nextEntries);
      setSubscription(nextSubscription);
      setPlans(nextPlans);
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

  useEffect(() => {
    const result = new URLSearchParams(window.location.search).get("billing");
    if (result === "success") setMessage("Checkout completed. Plan access updates after the verified billing webhook arrives.");
    if (result === "cancelled") setMessage("Checkout was cancelled. No plan change was applied.");
  }, []);

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

  async function beginCheckout(plan: BillingPlan) {
    setBusy(true);
    setError("");
    try {
      const hosted = await createSubscriptionCheckout(workspaceId, plan.key);
      window.location.assign(hosted.url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Checkout could not be opened.");
      setBusy(false);
    }
  }

  async function openPortal() {
    setBusy(true);
    setError("");
    try {
      const hosted = await createSubscriptionPortal(workspaceId);
      window.location.assign(hosted.url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Billing portal could not be opened.");
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
          <p>Role: {workspace.role} · account security, team access, plan limits, credits, privacy, and audit history.</p>
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

      <section className="panel operation-card subscription-card full-width">
        <div className="subscription-heading">
          <div>
            <div className="eyebrow">SUBSCRIPTION</div>
            <h2>{subscription?.plan.name ?? "Starter"}</h2>
            <p>
              {subscription?.subscription
                ? `${subscription.subscription.status.replaceAll("_", " ")}${subscription.subscription.cancel_at_period_end ? " · cancels at period end" : ""}`
                : "No paid subscription attached."}
            </p>
          </div>
          {canManageBilling && subscription?.portal_available && (
            <button className="secondary" type="button" disabled={busy} onClick={() => void openPortal()}>Manage billing</button>
          )}
        </div>
        {subscription?.plan && <strong className="plan-summary">{planDetail(subscription.plan)}</strong>}
        <div className="plan-grid">
          {plans.map((plan) => {
            const current = subscription?.plan.key === plan.key;
            return (
              <article className={`plan-card ${current ? "current" : ""}`} key={plan.key}>
                <div><span>{plan.name}</span>{current && <small>Current</small>}</div>
                <p>{plan.description}</p>
                <strong>{planDetail(plan)}</strong>
                <small>{number(plan.monthly_credits)} credits per paid invoice</small>
                {canManageBilling && !current && plan.key !== "starter" && (
                  <button className="primary" type="button" disabled={busy || !plan.checkout_available} onClick={() => void beginCheckout(plan)}>
                    {plan.checkout_available ? `Choose ${plan.name}` : "Price not configured"}
                  </button>
                )}
              </article>
            );
          })}
        </div>
      </section>

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
            <p>{subscription?.plan ? `${subscription.plan.max_members} seats included on ${subscription.plan.name}.` : "Seat limits follow the active plan."}</p>
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

      <PrivacyPanel workspaceId={workspaceId} workspaceSlug={workspace.slug} role={workspace.role} />

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
