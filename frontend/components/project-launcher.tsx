"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useMemo, useState } from "react";
import { createProject, startProject, uploadAssetResumable } from "@/lib/api";
import type { CameraMode, DialogueProtection, MusicDirectionMode } from "@/lib/types";

function splitRules(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function ProjectLauncher({ workspaceId }: { workspaceId: string }) {
  const router = useRouter();
  const [objective, setObjective] = useState("Turn these clips into a clear 45-second video with a strong hook and proof.");
  const [audience, setAudience] = useState("Prospective customers");
  const [platform, setPlatform] = useState("instagram_reels");
  const [duration, setDuration] = useState(45);
  const [cameraMode, setCameraMode] = useState<CameraMode>("required");
  const [threshold, setThreshold] = useState(0.72);
  const [musicDirectionMode, setMusicDirectionMode] = useState<MusicDirectionMode>("balanced");
  const [dialogueProtection, setDialogueProtection] = useState<DialogueProtection>("automatic");
  const [mustInclude, setMustInclude] = useState("proof, call to action");
  const [mustAvoid, setMustAvoid] = useState("private customer information");
  const [files, setFiles] = useState<File[]>([]);
  const [musicFile, setMusicFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");

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
        workspaceId,
        objective,
        audience,
        platform,
        duration,
        cameraMode,
        readinessThreshold: threshold,
        musicDirectionMode,
        dialogueProtection,
        mustInclude: splitRules(mustInclude),
        mustAvoid: splitRules(mustAvoid),
      });
      for (const [index, file] of files.entries()) {
        await uploadAssetResumable(project.id, file, "source_video", (fraction) => {
          setProgress(
            `Uploading source ${index + 1} of ${files.length}: ${file.name} · ${Math.round(fraction * 100)}%`,
          );
        });
      }
      if (musicFile) {
        await uploadAssetResumable(project.id, musicFile, "music", (fraction) => {
          setProgress(`Uploading music: ${musicFile.name} · ${Math.round(fraction * 100)}%`);
        });
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
    <section className="launch-shell">
      <header className="hero compact-hero">
        <div className="eyebrow">NEW PRODUCTION</div>
        <h1>Give the director footage and an outcome.</h1>
        <p>Uploads are chunked and resumable. Director OS audits what you shot and requests precise pickups when the story is incomplete.</p>
      </header>

      <form className="panel launch-form" onSubmit={submit}>
        <section className="form-section">
          <div className="section-heading"><span>01</span><div><h2>Outcome</h2><p>Define success before the timeline exists.</p></div></div>
          <label>Objective<textarea value={objective} onChange={(event) => setObjective(event.target.value)} required minLength={3} /></label>
          <div className="grid two">
            <label>Audience<input value={audience} onChange={(event) => setAudience(event.target.value)} /></label>
            <label>Platform<select value={platform} onChange={(event) => setPlatform(event.target.value)}><option value="instagram_reels">Instagram Reels</option><option value="tiktok">TikTok</option><option value="youtube_shorts">YouTube Shorts</option><option value="linkedin">LinkedIn</option></select></label>
            <label>Target duration<input type="number" min={5} max={600} value={duration} onChange={(event) => setDuration(Number(event.target.value))} /></label>
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
          <div className="section-heading"><span>03</span><div><h2>Music direction</h2><p>Choose the emotional range; Director OS still protects speech and musical timing.</p></div></div>
          <div className="choice-row">
            {(["restrained", "balanced", "expressive"] as MusicDirectionMode[]).map((mode) => (
              <button className={`choice ${musicDirectionMode === mode ? "active" : ""}`} type="button" key={mode} onClick={() => setMusicDirectionMode(mode)}><strong>{mode}</strong><small>{mode === "restrained" ? "Subtle lifts, rare accents" : mode === "balanced" ? "Natural dynamics and transitions" : "Bolder section lifts and accents"}</small></button>
            ))}
          </div>
          <div className="choice-row two">
            {(["automatic", "strong"] as DialogueProtection[]).map((mode) => (
              <button className={`choice ${dialogueProtection === mode ? "active" : ""}`} type="button" key={mode} onClick={() => setDialogueProtection(mode)}><strong>{mode} dialogue protection</strong><small>{mode === "automatic" ? "Adapt to detected speech" : "Keep music further below narration"}</small></button>
            ))}
          </div>
        </section>

        <section className="form-section">
          <div className="section-heading"><span>04</span><div><h2>Rules</h2><p>Comma or line separated. Current rules always beat learned preferences.</p></div></div>
          <div className="grid two"><label>Must include<textarea value={mustInclude} onChange={(event) => setMustInclude(event.target.value)} /></label><label>Must avoid<textarea value={mustAvoid} onChange={(event) => setMustAvoid(event.target.value)} /></label></div>
        </section>

        <section className="form-section">
          <div className="section-heading"><span>05</span><div><h2>Footage & music</h2><p>Large files resume from the last server-confirmed byte. Music is optional and must be yours to use.</p></div></div>
          <label className="drop-zone"><input type="file" accept="video/*" multiple onChange={(event) => setFiles(Array.from(event.target.files ?? []))} /><strong>Choose source videos</strong><span>{files.length ? `${files.length} file(s), ${(totalSize / 1024 / 1024).toFixed(1)} MB` : "MP4, MOV, WebM, or another browser-supported video"}</span></label>
          {files.length > 0 && <ul className="file-list">{files.map((file) => <li key={`${file.name}-${file.size}`}><span>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></li>)}</ul>}
          <label className="drop-zone music-zone"><input type="file" accept="audio/*" onChange={(event) => setMusicFile(event.target.files?.[0] ?? null)} /><strong>Optional music track</strong><span>{musicFile ? `${musicFile.name} · ${(musicFile.size / 1024 / 1024).toFixed(1)} MB` : "MP3, WAV, M4A, or another browser-supported audio file"}</span></label>
        </section>

        {error && <div className="alert error">{error}</div>}
        {progress && <div className="alert progress">{progress}</div>}
        <button className="primary action" disabled={busy} type="submit">{busy ? "Directing…" : "Start production"}</button>
      </form>
    </section>
  );
}
