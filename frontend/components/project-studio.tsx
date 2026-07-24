"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AuthPanel } from "@/components/auth-panel";
import { useAuth } from "@/components/auth-provider";
import { CaptureStudio } from "@/components/capture-studio";
import { RevisionChat } from "@/components/revision-chat";
import { getDirectorCamera, getProject, overrideDirectorCamera, resumeDirectorCamera } from "@/lib/api";
import type { CameraDimension, DirectorCamera, PickupMission, Project } from "@/lib/types";

const PIPELINE = ["queued", "analyzing", "needs_pickups", "planning", "rendering", "quality_check", "ready"] as const;

function dimensions(camera: DirectorCamera | null): Array<[string, CameraDimension]> {
  const value = camera?.report?.dimensions;
  if (!value) return [];
  if (Array.isArray(value)) return value.map((item, index) => [item.name ?? `Dimension ${index + 1}`, item]);
  return Object.entries(value);
}

function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

export function ProjectStudio() {
  const { session, loading: authLoading } = useAuth();
  const params = useParams<{ projectId: string }>();
  const projectId = params.projectId;
  const [project, setProject] = useState<Project | null>(null);
  const [camera, setCamera] = useState<DirectorCamera | null>(null);
  const [selectedMission, setSelectedMission] = useState<PickupMission | null>(null);
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    if (!session) return;
    try {
      const [nextProject, nextCamera] = await Promise.all([getProject(projectId), getDirectorCamera(projectId)]);
      setProject(nextProject);
      setCamera(nextCamera);
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load project.");
    } finally {
      setLoading(false);
    }
  }, [projectId, session]);

  useEffect(() => {
    if (!session) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3500);
    return () => window.clearInterval(timer);
  }, [refresh, session]);

  const progressIndex = useMemo(() => Math.max(0, PIPELINE.indexOf((project?.status ?? "queued") as (typeof PIPELINE)[number])), [project?.status]);
  const openMissions = camera?.missions.filter((mission) => ["requested", "submitted", "rejected"].includes(mission.status)) ?? [];

  async function resume() {
    setAction("Resuming Director Camera validation…");
    try { await resumeDirectorCamera(projectId); await refresh(); } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not resume."); } finally { setAction(""); }
  }

  async function override() {
    const reason = window.prompt("Why should Director OS continue without the requested pickup?");
    if (!reason?.trim()) return;
    setAction("Recording override and resuming…");
    try { await overrideDirectorCamera(projectId, reason.trim()); await refresh(); } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not override."); } finally { setAction(""); }
  }

  if (authLoading) return <main className="shell"><div className="panel loading">Opening secure workspace…</div></main>;
  if (!session) return <AuthPanel />;
  if (loading) return <main className="shell"><div className="panel loading">Loading Director OS…</div></main>;
  if (!project) return <main className="shell"><div className="panel"><h1>Project unavailable</h1><p>{error}</p><Link href="/">Return to workspace</Link></div></main>;

  return (
    <main className="shell studio-shell">
      <header className="studio-header">
        <div><Link className="back" href="/">← Production library</Link><div className="eyebrow">PROJECT {project.id.slice(0, 8)}</div><h1>{project.contract.objective}</h1></div>
        <div className={`status-pill status-${project.status}`}>{statusLabel(project.status)}</div>
      </header>

      <section className="pipeline panel" aria-label="Production progress">
        {PIPELINE.map((step, index) => <div key={step} className={`pipeline-step ${index <= progressIndex ? "complete" : ""} ${project.status === step ? "current" : ""}`}><span>{index + 1}</span><small>{statusLabel(step)}</small></div>)}
      </section>

      {error && <div className="alert error">{error}</div>}
      {action && <div className="alert progress">{action}</div>}

      <div className="studio-grid">
        <section className="panel readiness-panel">
          <div className="panel-title"><div><div className="eyebrow">PRODUCTION READINESS</div><h2>{camera?.readiness_score == null ? "Audit pending" : `${Math.round(camera.readiness_score * 100)}% ready`}</h2></div>{camera?.readiness_score != null && <div className="readiness-ring" style={{ "--score": `${camera.readiness_score * 360}deg` } as React.CSSProperties}><span>{Math.round(camera.readiness_score * 100)}</span></div>}</div>
          <p className="muted">Mode: <strong>{camera?.mode ?? project.contract.director_camera_mode}</strong> · threshold {Math.round((camera?.threshold ?? project.contract.production_readiness_threshold) * 100)}%</p>
          <div className="dimension-list">{dimensions(camera).map(([name, item]) => <article key={name} className={item.blocking ? "dimension blocking" : "dimension"}><div><strong>{name.replaceAll("_", " ")}</strong><span>{Math.round((item.score ?? 0) * 100)}%</span></div><div className="meter"><i style={{ width: `${Math.round((item.score ?? 0) * 100)}%` }} /></div>{item.findings?.map((finding) => <small key={finding}>{finding}</small>)}</article>)}</div>
        </section>

        <section className="panel">
          <div className="panel-title"><div><div className="eyebrow">ASSETS</div><h2>{project.assets.length} uploaded</h2></div></div>
          <ul className="asset-list">{project.assets.map((asset) => <li key={asset.id}><span><strong>{asset.original_filename}</strong><small>{asset.kind.replaceAll("_", " ")}</small></span><small>{(asset.size_bytes / 1024 / 1024).toFixed(1)} MB</small></li>)}</ul>
          {project.output_available && <div className="alert success">A publishable output is ready for secure preview or download.</div>}
          {project.error_message && <div className="alert error">{project.error_message}</div>}
        </section>
      </div>

      <section className="panel mission-section">
        <div className="panel-title"><div><div className="eyebrow">DIRECTOR CAMERA MISSIONS</div><h2>{openMissions.length ? `${openMissions.length} action${openMissions.length === 1 ? "" : "s"} needed` : "No open pickups"}</h2></div>{project.status === "needs_pickups" && <div className="inline-actions"><button className="secondary" onClick={override}>Override gate</button><button className="primary" onClick={resume}>Validate & resume</button></div>}</div>
        <div className="mission-grid">{camera?.missions.map((mission) => <article className={`mission-card mission-${mission.status}`} key={mission.id}><div className="mission-top"><span className={`priority priority-${mission.priority}`}>{mission.priority}</span><span>{mission.status}</span></div><h3>{mission.title}</h3><p>{mission.reason}</p><div className="mission-meta"><span>{mission.mission_type.replaceAll("_", " ")}</span>{mission.target_terms.length > 0 && <span>{mission.target_terms.slice(0, 3).join(" · ")}</span>}</div>{mission.error_message && <small className="error-text">{mission.error_message}</small>}{["requested", "rejected"].includes(mission.status) && <button className="primary" onClick={() => setSelectedMission(mission)}>Open guided capture</button>}</article>)}</div>
      </section>

      <RevisionChat projectId={projectId} outputAvailable={project.output_available} />

      {selectedMission && <CaptureStudio projectId={projectId} mission={selectedMission} onClose={() => setSelectedMission(null)} onSubmitted={async () => { setSelectedMission(null); await refresh(); }} />}
    </main>
  );
}
