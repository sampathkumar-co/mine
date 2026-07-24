import type {
  CameraMode,
  DirectorCamera,
  Project,
  ProjectAccepted,
} from "@/lib/types";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export interface CreateProjectInput {
  userId: string;
  objective: string;
  audience: string;
  platform: string;
  duration: number;
  cameraMode: CameraMode;
  readinessThreshold: number;
  mustInclude: string[];
  mustAvoid: string[];
}

export async function createProject(input: CreateProjectInput): Promise<Project> {
  return request<Project>("/projects", {
    method: "POST",
    body: JSON.stringify({
      user_id: input.userId,
      contract: {
        objective: input.objective,
        target_audience: input.audience || null,
        target_platform: input.platform,
        target_duration_seconds: input.duration,
        tier: 1,
        instructions: null,
        must_include: input.mustInclude,
        must_avoid: input.mustAvoid,
        reference_rules: {},
        brand_rules: {},
        creative_freedom: 0.6,
        director_profile_key: "default",
        use_director_memory: true,
        director_camera_mode: input.cameraMode,
        production_readiness_threshold: input.readinessThreshold,
      },
    }),
  });
}

export async function uploadAsset(projectId: string, file: File, kind = "source_video"): Promise<void> {
  const form = new FormData();
  form.append("kind", kind);
  form.append("file", file);
  await request(`/projects/${projectId}/assets`, { method: "POST", body: form });
}

export function startProject(projectId: string): Promise<ProjectAccepted> {
  return request<ProjectAccepted>(`/projects/${projectId}/start`, { method: "POST" });
}

export function getProject(projectId: string): Promise<Project> {
  return request<Project>(`/projects/${projectId}`);
}

export function getDirectorCamera(projectId: string): Promise<DirectorCamera> {
  return request<DirectorCamera>(`/projects/${projectId}/director-camera`);
}

export async function submitPickup(projectId: string, missionId: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  await request(`/projects/${projectId}/director-camera/missions/${missionId}/submit`, {
    method: "POST",
    body: form,
  });
}

export function resumeDirectorCamera(projectId: string): Promise<ProjectAccepted> {
  return request<ProjectAccepted>(`/projects/${projectId}/director-camera/resume`, { method: "POST" });
}

export function overrideDirectorCamera(projectId: string, reason: string): Promise<ProjectAccepted> {
  return request<ProjectAccepted>(`/projects/${projectId}/director-camera/override`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}
