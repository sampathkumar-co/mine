"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { createDeliveryLink, createRevision, listRevisions } from "@/lib/api";
import type { RevisionSummary } from "@/lib/types";

export function RevisionChat({ projectId, outputAvailable }: { projectId: string; outputAvailable: boolean }) {
  const [revisions, setRevisions] = useState<RevisionSummary[]>([]);
  const [instruction, setInstruction] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    if (!outputAvailable) return;
    try {
      setRevisions(await listRevisions(projectId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load revisions.");
    }
  }, [outputAvailable, projectId]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!instruction.trim()) return;
    setBusy(true);
    setError("");
    try {
      await createRevision(projectId, instruction.trim());
      setInstruction("");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Revision could not be queued.");
    } finally {
      setBusy(false);
    }
  }

  async function preview() {
    try {
      const link = await createDeliveryLink(projectId);
      setPreviewUrl(link.url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Preview could not be opened.");
    }
  }

  async function download() {
    try {
      const link = await createDeliveryLink(projectId, { download: true });
      window.location.assign(link.url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Download could not be prepared.");
    }
  }

  return (
    <section className="panel revision-panel">
      <div className="panel-title">
        <div><div className="eyebrow">DELIVERY & REVISION CHAT</div><h2>{outputAvailable ? "Shape the finished cut" : "Available after the first render"}</h2></div>
        {outputAvailable && <div className="inline-actions"><button className="secondary" type="button" onClick={preview}>Secure preview</button><button className="primary" type="button" onClick={download}>Download</button></div>}
      </div>

      {previewUrl && <video className="delivery-preview" src={previewUrl} controls playsInline />}

      <div className="revision-thread">
        {revisions.length ? revisions.slice().reverse().map((revision) => (
          <article key={revision.version} className={`revision-message ${revision.is_active ? "active" : ""}`}>
            <div><strong>Version {revision.version}</strong><span>{revision.status}</span></div>
            <p>{revision.instruction ?? "Autonomous production"}</p>
            {revision.error_message && <small className="error-text">{revision.error_message}</small>}
          </article>
        )) : <p className="muted">No revision history yet.</p>}
      </div>

      <form className="revision-form" onSubmit={submit}>
        <textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="Try: Shorten the intro, make captions larger, and use less B-roll." disabled={!outputAvailable || busy} />
        <button className="primary" disabled={!outputAvailable || busy || !instruction.trim()} type="submit">{busy ? "Queuing…" : "Revise video"}</button>
      </form>
      {error && <div className="alert error">{error}</div>}
      <p className="muted delivery-note">Preview and download URLs expire automatically and never reveal the server storage path.</p>
    </section>
  );
}
