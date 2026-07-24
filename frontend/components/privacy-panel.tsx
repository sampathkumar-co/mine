"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  cancelWorkspaceDeletion,
  createPrivacyDelivery,
  createWorkspaceExport,
  listPrivacyRequests,
  resolvePrivacyDeliveryUrl,
  scheduleWorkspaceDeletion,
} from "@/lib/governance-api";
import type { PrivacyRequest } from "@/lib/governance-types";
import type { WorkspaceRole } from "@/lib/types";

function requestLabel(request: PrivacyRequest): string {
  if (request.kind === "export") {
    if (request.status === "ready" && request.available_until) {
      return `Ready until ${new Date(request.available_until).toLocaleString()}`;
    }
    return request.status.replaceAll("_", " ");
  }
  if (request.status === "scheduled" && request.execute_after) {
    return `Scheduled for ${new Date(request.execute_after).toLocaleString()}`;
  }
  return request.status.replaceAll("_", " ");
}

export function PrivacyPanel({
  workspaceId,
  workspaceSlug,
  role,
}: {
  workspaceId: string;
  workspaceSlug: string;
  role: WorkspaceRole;
}) {
  const [requests, setRequests] = useState<PrivacyRequest[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [reason, setReason] = useState("");
  const canManage = role === "owner" || role === "admin";
  const canDelete = role === "owner";

  const load = useCallback(async () => {
    if (!canManage) return;
    try {
      setRequests(await listPrivacyRequests(workspaceId));
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Privacy requests could not be loaded.");
    }
  }, [canManage, workspaceId]);

  useEffect(() => {
    void load();
  }, [load]);

  const scheduledDeletion = useMemo(
    () => requests.find((item) => item.kind === "deletion" && item.status === "scheduled"),
    [requests],
  );

  async function exportWorkspace() {
    setBusy(true);
    setError("");
    try {
      const created = await createWorkspaceExport(workspaceId);
      setMessage(created.status === "ready" ? "Workspace export is ready." : "Workspace export was queued.");
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Workspace export failed.");
    } finally {
      setBusy(false);
    }
  }

  async function download(request: PrivacyRequest) {
    setBusy(true);
    setError("");
    try {
      const delivery = await createPrivacyDelivery(workspaceId, request.id);
      window.location.assign(resolvePrivacyDeliveryUrl(delivery.url));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Export download could not be created.");
      setBusy(false);
    }
  }

  async function scheduleDeletion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const created = await scheduleWorkspaceDeletion(workspaceId, confirmation, reason);
      setMessage(`Workspace deletion is scheduled for ${created.execute_after ? new Date(created.execute_after).toLocaleString() : "the end of the grace period"}.`);
      setConfirmation("");
      setReason("");
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Deletion could not be scheduled.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelDeletion(request: PrivacyRequest) {
    setBusy(true);
    setError("");
    try {
      await cancelWorkspaceDeletion(workspaceId, request.id);
      setMessage("Workspace deletion was cancelled.");
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Deletion could not be cancelled.");
    } finally {
      setBusy(false);
    }
  }

  if (!canManage) return null;

  return (
    <section className="panel operation-card full-width privacy-card">
      <div className="eyebrow">DATA & PRIVACY</div>
      <div className="privacy-heading">
        <div>
          <h2>Export or close this workspace</h2>
          <p>Exports contain workspace metadata, editorial decisions, usage records, and a content manifest. Raw media stays in its existing secure delivery path.</p>
        </div>
        <button className="secondary" type="button" disabled={busy} onClick={() => void exportWorkspace()}>
          Create export
        </button>
      </div>
      {message && <div className="alert success">{message}</div>}
      {error && <div className="alert error">{error}</div>}
      <div className="privacy-request-list">
        {requests.slice(0, 8).map((request) => (
          <article key={request.id}>
            <div>
              <strong>{request.kind === "export" ? "Workspace export" : "Workspace deletion"}</strong>
              <small>{requestLabel(request)}</small>
            </div>
            {request.kind === "export" && request.status === "ready" && (
              <button className="text-button" type="button" disabled={busy} onClick={() => void download(request)}>Download</button>
            )}
            {request.kind === "deletion" && request.status === "scheduled" && canDelete && (
              <button className="secondary" type="button" disabled={busy} onClick={() => void cancelDeletion(request)}>Cancel deletion</button>
            )}
          </article>
        ))}
      </div>
      {canDelete && !scheduledDeletion && (
        <form className="privacy-delete-form" onSubmit={scheduleDeletion}>
          <div>
            <h3>Schedule permanent deletion</h3>
            <p>Paid subscriptions must be cancelled and active production must finish first. New mutations lock during the grace period.</p>
          </div>
          <label>
            Reason
            <input value={reason} onChange={(event) => setReason(event.target.value)} minLength={3} required placeholder="Why is this workspace being closed?" />
          </label>
          <label>
            Type <code>{workspaceSlug}</code> to confirm
            <input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} required autoComplete="off" />
          </label>
          <button className="danger" disabled={busy || confirmation !== workspaceSlug}>Schedule deletion</button>
        </form>
      )}
    </section>
  );
}
