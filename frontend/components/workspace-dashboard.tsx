"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { AuthPanel } from "@/components/auth-panel";
import { useAuth } from "@/components/auth-provider";
import { ProjectLauncher } from "@/components/project-launcher";
import { createWorkspace, listWorkspaceProjects, requestEmailVerification } from "@/lib/api";
import type { WorkspaceProject } from "@/lib/types";

function statusLabel(value: string): string {
  return value.replaceAll("_", " ");
}

export function WorkspaceDashboard() {
  const { session, loading, logout, refresh } = useAuth();
  const [workspaceId, setWorkspaceId] = useState("");
  const [projects, setProjects] = useState<WorkspaceProject[]>([]);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!session?.workspaces.length) return;
    const stored = window.localStorage.getItem("director-workspace-id");
    const selected = session.workspaces.some((workspace) => workspace.id === stored)
      ? stored ?? session.workspaces[0].id
      : session.workspaces[0].id;
    setWorkspaceId(selected);
  }, [session]);

  useEffect(() => {
    if (!workspaceId) return;
    window.localStorage.setItem("director-workspace-id", workspaceId);
    setLibraryLoading(true);
    listWorkspaceProjects(workspaceId)
      .then((items) => {
        setProjects(items);
        setError("");
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Could not load productions."))
      .finally(() => setLibraryLoading(false));
  }, [workspaceId]);

  const workspace = useMemo(
    () => session?.workspaces.find((item) => item.id === workspaceId) ?? session?.workspaces[0],
    [session, workspaceId],
  );

  async function addWorkspace() {
    const name = window.prompt("Name the new workspace");
    if (!name?.trim()) return;
    try {
      const created = await createWorkspace(name.trim());
      await refresh();
      setWorkspaceId(created.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create workspace.");
    }
  }

  async function sendVerification() {
    try {
      const response = await requestEmailVerification();
      setMessage(response.message);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not request verification.");
    }
  }

  if (loading) return <main className="shell"><div className="panel loading">Opening Director OS…</div></main>;
  if (!session) return <AuthPanel />;

  return (
    <main className="shell workspace-shell">
      <header className="workspace-bar">
        <div>
          <div className="eyebrow">DIRECTOR OS / WORKSPACE</div>
          <h1>{workspace?.name ?? "Production workspace"}</h1>
          <p>Signed in as {session.user.display_name} · {session.user.email}</p>
        </div>
        <div className="workspace-actions">
          <select value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)} aria-label="Workspace">
            {session.workspaces.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          {workspace && <Link className="secondary button-link" href={`/workspaces/${workspace.id}/settings`}>Settings</Link>}
          <button className="secondary" type="button" onClick={addWorkspace}>New workspace</button>
          <button className="secondary" type="button" onClick={() => void logout()}>Sign out</button>
        </div>
      </header>

      {!session.user.email_verified && (
        <div className="alert progress verification-banner">
          <div><strong>Email verification pending</strong><p>Verify your address to secure recovery and team invitations.</p></div>
          <button className="primary" type="button" onClick={() => void sendVerification()}>Send verification</button>
        </div>
      )}
      {message && <div className="alert success">{message}</div>}
      {error && <div className="alert error">{error}</div>}

      <section className="library-section">
        <div className="panel-title">
          <div><div className="eyebrow">PRODUCTION LIBRARY</div><h2>{projects.length} production{projects.length === 1 ? "" : "s"}</h2></div>
          {workspace && <span className="status-pill">{workspace.role}</span>}
        </div>
        {libraryLoading ? (
          <div className="panel loading">Loading productions…</div>
        ) : projects.length ? (
          <div className="project-library">
            {projects.map((project) => (
              <Link href={`/projects/${project.id}`} className="project-card panel" key={project.id}>
                <div className="mission-top"><span>{project.target_platform.replaceAll("_", " ")}</span><span className={`status-pill status-${project.status}`}>{statusLabel(project.status)}</span></div>
                <h3>{project.objective}</h3>
                <p>{project.target_duration_seconds}s target · {project.asset_count} asset{project.asset_count === 1 ? "" : "s"}</p>
                <small>{project.output_available ? "Publishable output ready" : "Production in progress"}</small>
              </Link>
            ))}
          </div>
        ) : (
          <div className="panel empty-library"><h3>No productions yet</h3><p>Your first Director Contract will appear here.</p></div>
        )}
      </section>

      {workspaceId && workspace?.role !== "viewer" && <ProjectLauncher workspaceId={workspaceId} />}
      {workspace?.role === "viewer" && <div className="panel empty-library"><h3>Viewer access</h3><p>You can inspect projects and outputs, while editors manage production changes.</p></div>}
    </main>
  );
}
