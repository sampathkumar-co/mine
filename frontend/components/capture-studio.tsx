"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getGhostFrame, submitPickup } from "@/lib/api";
import type { PickupMission } from "@/lib/types";

interface CaptureStudioProps {
  projectId: string;
  mission: PickupMission;
  onClose: () => void;
  onSubmitted: () => Promise<void>;
}

interface CaptureSignals {
  audio: number;
  light: number;
  stability: number;
  level: number | null;
}

function bestMimeType(): string {
  const types = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

function guidance(specification: Record<string, unknown>): string[] {
  return Object.entries(specification)
    .filter(([, value]) => typeof value === "string" || typeof value === "number" || typeof value === "boolean")
    .slice(0, 8)
    .map(([key, value]) => `${key.replaceAll("_", " ")}: ${String(value)}`);
}

export function CaptureStudio({ projectId, mission, onClose, onSubmitted }: CaptureStudioProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const [signals, setSignals] = useState<CaptureSignals>({ audio: 0, light: 0.5, stability: 1, level: null });
  const [ghost, setGhost] = useState<string | null>(null);
  const [ghostLabel, setGhostLabel] = useState("Loading continuity reference…");
  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [permission, setPermission] = useState("Requesting camera and microphone…");

  const stopStream = useCallback(() => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioContextRef.current?.close();
    audioContextRef.current = null;
  }, []);

  useEffect(() => {
    let objectUrl = "";
    let disposed = false;
    getGhostFrame(projectId, mission.id)
      .then((blob) => {
        if (disposed) return;
        objectUrl = URL.createObjectURL(blob);
        setGhost(objectUrl);
        setGhostLabel("Project continuity frame");
      })
      .catch(() => setGhostLabel("No project continuity frame available"));
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [mission.id, projectId]);

  useEffect(() => {
    let disposed = false;
    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1080 }, height: { ideal: 1920 } },
          audio: { echoCancellation: true, noiseSuppression: true },
        });
        if (disposed) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setPermission("");

        const context = new AudioContext();
        audioContextRef.current = context;
        const source = context.createMediaStreamSource(stream);
        const analyser = context.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        const samples = new Uint8Array(analyser.frequencyBinCount);
        let previousBrightness = 0.5;
        let stability = 1;

        const measure = () => {
          analyser.getByteTimeDomainData(samples);
          const rms = Math.sqrt(
            samples.reduce((sum, sample) => sum + ((sample - 128) / 128) ** 2, 0) /
              samples.length,
          );
          const video = videoRef.current;
          const canvas = canvasRef.current;
          let light = previousBrightness;
          if (video && canvas && video.videoWidth > 0) {
            const context2d = canvas.getContext("2d", { willReadFrequently: true });
            if (context2d) {
              canvas.width = 32;
              canvas.height = 32;
              context2d.drawImage(video, 0, 0, 32, 32);
              const pixels = context2d.getImageData(0, 0, 32, 32).data;
              let total = 0;
              for (let index = 0; index < pixels.length; index += 4) {
                total += (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
              }
              light = total / (pixels.length / 4) / 255;
              const delta = Math.abs(light - previousBrightness);
              stability = Math.max(
                0,
                Math.min(1, stability * 0.85 + (1 - Math.min(1, delta * 10)) * 0.15),
              );
              previousBrightness = light;
            }
          }
          setSignals((current) => ({
            ...current,
            audio: Math.min(1, rms * 6),
            light,
            stability,
          }));
          animationRef.current = requestAnimationFrame(measure);
        };
        measure();
      } catch (caught) {
        setPermission("");
        setError(caught instanceof Error ? caught.message : "Camera permission was not granted.");
      }
    }
    void start();

    const orientation = (event: DeviceOrientationEvent) =>
      setSignals((current) => ({
        ...current,
        level: event.gamma == null ? null : Math.max(-45, Math.min(45, event.gamma)),
      }));
    window.addEventListener("deviceorientation", orientation);
    return () => {
      disposed = true;
      window.removeEventListener("deviceorientation", orientation);
      stopStream();
    };
  }, [stopStream]);

  useEffect(() => {
    if (!recording) return;
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  function setLocalGhostFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    setGhost(canvas.toDataURL("image/jpeg", 0.72));
    setGhostLabel("Frozen live alignment frame");
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) return;
    chunksRef.current = [];
    setRecorded(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setElapsed(0);
    const mimeType = bestMimeType();
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType, videoBitsPerSecond: 5_000_000 } : undefined,
    );
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
      const file = new File([blob], `pickup-${mission.id}.webm`, {
        type: blob.type,
        lastModified: Date.now(),
      });
      setRecorded(file);
      setPreview(URL.createObjectURL(blob));
    };
    recorder.start(500);
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function submit() {
    if (!recorded) return;
    setSubmitting(true);
    setError("");
    try {
      await submitPickup(projectId, mission.id, recorded);
      await onSubmitted();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Pickup upload failed.");
    } finally {
      setSubmitting(false);
    }
  }

  const lightLabel = signals.light < 0.24 ? "Too dark" : signals.light > 0.9 ? "Highlights clipping" : "Light usable";
  const audioLabel = signals.audio < 0.04 ? "Speak to test audio" : signals.audio > 0.88 ? "Audio peaking" : "Audio usable";
  const levelLabel = signals.level == null ? "Level unavailable" : Math.abs(signals.level) < 4 ? "Camera level" : `Tilt ${Math.round(signals.level)}°`;

  return (
    <div className="capture-backdrop" role="dialog" aria-modal="true" aria-label={`Guided capture: ${mission.title}`}>
      <section className="capture-modal">
        <header>
          <div>
            <div className="eyebrow">GUIDED PICKUP / {mission.mission_type.replaceAll("_", " ")}</div>
            <h2>{mission.title}</h2>
            <p>{mission.reason}</p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close capture">×</button>
        </header>
        <div className="capture-grid">
          <div className="camera-stage">
            <video ref={videoRef} muted playsInline />
            {ghost && <img className="ghost-frame" src={ghost} alt="Ghost alignment reference" />}
            <div className="ghost-label">{ghostLabel}</div>
            <div className="safe-zone"><span>SAFE</span></div>
            <div className="thirds vertical-one" /><div className="thirds vertical-two" />
            <div className="thirds horizontal-one" /><div className="thirds horizontal-two" />
            <div className="eye-line">EYE LINE</div>
            {recording && <div className="record-badge"><i /> REC {elapsed.toFixed(1)}s</div>}
            {permission && <div className="camera-message">{permission}</div>}
          </div>
          <aside className="capture-controls">
            <div className="signal-grid">
              <div className={signals.light < 0.24 || signals.light > 0.9 ? "signal warn" : "signal good"}><span>LIGHT</span><strong>{lightLabel}</strong><div className="meter"><i style={{ width: `${signals.light * 100}%` }} /></div></div>
              <div className={signals.audio > 0.88 ? "signal warn" : "signal good"}><span>AUDIO</span><strong>{audioLabel}</strong><div className="meter"><i style={{ width: `${signals.audio * 100}%` }} /></div></div>
              <div className={signals.stability < 0.75 ? "signal warn" : "signal good"}><span>STABILITY</span><strong>{Math.round(signals.stability * 100)}%</strong><div className="meter"><i style={{ width: `${signals.stability * 100}%` }} /></div></div>
              <div className={signals.level != null && Math.abs(signals.level) >= 4 ? "signal warn" : "signal good"}><span>LEVEL</span><strong>{levelLabel}</strong></div>
            </div>
            <div className="mission-brief"><h3>Shot brief</h3><ul>{guidance(mission.specification).map((item) => <li key={item}>{item}</li>)}{mission.target_terms.length > 0 && <li>Show or say: {mission.target_terms.join(", ")}</li>}</ul></div>
            <div className="capture-actions"><button className="secondary" onClick={setLocalGhostFrame}>Freeze live frame</button>{!recording ? <button className="record-button" onClick={startRecording} disabled={!streamRef.current}>Record</button> : <button className="stop-button" onClick={stopRecording}>Stop</button>}</div>
            {preview && <div className="take-review"><video src={preview} controls playsInline /><div><button className="secondary" onClick={startRecording}>Retake</button><button className="primary" onClick={submit} disabled={submitting}>{submitting ? "Uploading…" : "Submit pickup"}</button></div></div>}
            {error && <div className="alert error">{error}</div>}
          </aside>
        </div>
        <canvas ref={canvasRef} hidden />
      </section>
    </div>
  );
}
