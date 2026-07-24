"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { createProject, startProject, uploadAsset } from "@/lib/api";
import type { CameraMode } from "@/lib/types";

function splitRules(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function defaultUserId(): string {
  if (typeof window === "undefined") return "";
  const existing = window.localStorage.getItem("director-user-id");
  if (existing) return existing;
  const created = crypto.randomUUID();
  window.localStorage.setItem("director-user-id", created);
  return created;
}

export function ProjectLauncher() {
  const router = useRouter();
  const [userId, setUserId] = useState("");
  const [objective, setObjective] = useState("Turn these clips into a clear 45-second video with a strong hook and proof.");
  const [audience, setAudience] = useState("Prospective customers");
  const [platform, setPlatform] = useState("instagram_reels");
  const [duration, setDuration] = useState(45);
  const [cameraMode, setCameraMode] = useState<CameraMode>("required");
  const [threshold, setThreshold] = useState(0.72);
  const [mustInclude, setMustInclude] = useState("proof, call to action");
  const [mustAvoid, setMustAvoid] = useState("private customer information");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");

  useEffect(() => setUserId(defaultUserId()), []);
  const totalSize = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!files.length) {
      setError("Add at least one source video.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      setProgress("Creating the Director Contract…");
      const project = await createProject({
        userId,
        objective,
        audience,
        platform,
        duration,
        cameraMode,
        readinessThreshold: threshold,
        mustInclude: splitRules(mustInclude),
        mustAvoid: splitRules(mustAvoid),
      });
      for (const [index, file] of files.entries()) {
        setProgress(`Uploading source ${index + 1} of ${files.length}: ${file.name}`);
        await uploadAsset(project.id, file);
      }
      setProgress("Starting autonomous production…");
      await startProject(project.id);
      router.push(`/projects/${project.id}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Project creation failed.");
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  return (
    <main className="shell launch-shell">
      <header className="hero">
        <div className="eyebrow">DIRECTOR OS / PRODUCTION INTAKE</div>
        <h1>Give the director footage and an outcome.</h1>
        <p>Director OS audits what you shot, edits what is usable, and asks for precise pickups when the story is incomplete.</p>
      </header>

      <form className="panel launch-form" onSubmit={submit}>
        <section className="form-section">
          <div className="section-heading"><span>01</span><div><h2>Outcome</h2><p>Define success before the timeline exists.</p></div></div>
          <label>Objective<textarea value={objective} onChange={(event) => setObjective(event.target.value)} required minLength={3} /></label>
          <div className="grid two">
            <label>Audience<input value={audience} onChange={(event) => setAudience(event.target.value)} /></label>
            <label>Platform<select value={platform} onChange={(event) => setPlatform(event.target.value)}><option value="instagram_reels">Instagram Reels</option><option value="tiktok">TikTok</option><option value="youtube_shorts">YouTube Shorts</option><option value="linkedin">LinkedIn</option></select></label>
            <label>Target duration<input type="number" min={5} max={600} value={duration} onChange={(event) => setDuration(Number(event.target.value))} /></label>
            <label>User ID<input value={userId} onChange={(event) => setUserId(event.target.value)} required /></label>
          </div>
        </section>

        <section className="form-section">
          <div className="section-heading"><span>02</span><div><h2>Director Camera</h2><p>Choose whether missing shots should pause production.</p></div></div>
          <div className="choice-row">
            {(["off", "advisory", "required"] as CameraMode[]).map((mode) => (
              <button className={`choice ${cameraMode === mode ? "active" : ""}`} type="button" key={mode} onClick={() => setCameraMode(mode)}><strong>{mode}</strong><small>{mode === "off" ? "Edit what exists" : mode === "advisory" ? "Render and recommend" : "Pause for critical pickups"}</small></button>
            ))}
          </div>
          <label>Readiness threshold <output>{Math.round(threshold * 100)}%</output><input className="range" type="range" min="0.5" max="0.95" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>
        </section>

        <section className="form-section">
          <div className="section-heading"><span>03</span><div><h2>Rules</h2><p>Comma or line separated. Current rules always beat learned preferences.</p></div></div>
          <div className="grid two"><label>Must include<textarea value={mustInclude} onChange={(event) => setMustInclude(event.target.value)} /></label><label>Must avoid<textarea value={mustAvoid} onChange={(event) => setMustAvoid(event.target.value)} /></label></div>
        </section>

        <section className="form-section">
          <div className="section-heading"><span>04</span><div><h2>Footage</h2><p>Upload up to the backend-configured source limit.</p></div></div>
          <label className="drop-zone"><input type="file" accept="video/*" multiple onChange={(event) => setFiles(Array.from(event.target.files ?? []))} /><strong>Choose source videos</strong><span>{files.length ? `${files.length} file(s), ${(totalSize / 1024 / 1024).toFixed(1)} MB` : "MP4, MOV, WebM, or another browser-supported video"}</span></label>
          {files.length > 0 && <ul className="file-list">{files.map((file) => <li key={`${file.name}-${file.size}`}><span>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></li>)}</ul>}
        </section>

        {error && <div className="alert error">{error}</div>}
        {progress && <div className="alert progress">{progress}</div>}
        <button className="primary action" disabled={busy || !userId} type="submit">{busy ? "Directing…" : "Start production"}</button>
      </form>
    </main>
  );
}
