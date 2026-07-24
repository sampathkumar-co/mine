import type {
  AuthSession,
  CameraMode,
  DeliveryLink,
  DirectorCamera,
  Project,
  ProjectAccepted,
  ResumableUpload,
  RevisionAccepted,
  RevisionSummary,
  Workspace,
  WorkspaceProject,
} from "@/lib/types";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");
const TOKEN_KEY = "director-os-access-token";
const UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
  else window.sessionStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event("director-auth-changed"));
}

function authHeaders(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {
    // Keep the status-based message when the response is not JSON.
  }
  if (response.status === 401) setAccessToken(null);
  return new ApiError(message, response.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const hasStructuredBody = init?.body && !(init.body instanceof FormData) && !(init.body instanceof Blob);
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...authHeaders(),
      ...(hasStructuredBody ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

export async function register(input: {
  email: string;
  password: string;
  displayName: string;
  workspaceName: string;
}): Promise<AuthSession> {
  const session = await request<AuthSession>("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      email: input.email,
      password: input.password,
      display_name: input.displayName,
      workspace_name: input.workspaceName,
    }),
  });
  setAccessToken(session.access_token);
  return session;
}

export async function login(email: string, password: string): Promise<AuthSession> {
  const session = await request<AuthSession>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setAccessToken(session.access_token);
  return session;
}

export async function getSession(): Promise<AuthSession> {
  const session = await request<AuthSession>("/auth/me");
  setAccessToken(session.access_token);
  return session;
}

export function listWorkspaces(): Promise<Workspace[]> {
  return request<Workspace[]>("/workspaces");
}

export function createWorkspace(name: string): Promise<Workspace> {
  return request<Workspace>("/workspaces", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function listWorkspaceProjects(workspaceId: string): Promise<WorkspaceProject[]> {
  return request<WorkspaceProject[]>(`/workspaces/${workspaceId}/projects`);
}

export interface CreateProjectInput {
  workspaceId: string;
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
      workspace_id: input.workspaceId,
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

export async function uploadAssetResumable(
  projectId: string,
  file: File,
  kind = "source_video",
  onProgress?: (fraction: number) => void,
): Promise<void> {
  let upload = await request<ResumableUpload>(`/projects/${projectId}/uploads`, {
    method: "POST",
    body: JSON.stringify({
      kind,
      original_filename: file.name,
      content_type: file.type || "video/mp4",
      total_bytes: file.size,
    }),
  });

  while (upload.received_bytes < file.size) {
    const start = upload.received_bytes;
    const end = Math.min(file.size, start + UPLOAD_CHUNK_BYTES);
    const chunk = file.slice(start, end, "application/offset+octet-stream");
    try {
      upload = await request<ResumableUpload>(`/uploads/${upload.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/offset+octet-stream",
          "Upload-Offset": String(start),
        },
        body: chunk,
      });
    } catch (caught) {
      if (!(caught instanceof ApiError) || caught.status !== 409) throw caught;
      upload = await request<ResumableUpload>(`/uploads/${upload.id}`);
    }
    onProgress?.(Math.min(1, upload.received_bytes / file.size));
  }
}

export async function uploadAsset(projectId: string, file: File, kind = "source_video"): Promise<void> {
  await uploadAssetResumable(projectId, file, kind);
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

export async function getGhostFrame(projectId: string, missionId: string): Promise<Blob> {
  const response = await fetch(
    `${API_URL}/projects/${projectId}/director-camera/missions/${missionId}/ghost-frame`,
    { headers: authHeaders(), cache: "no-store" },
  );
  if (!response.ok) throw await responseError(response);
  return response.blob();
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

export function listRevisions(projectId: string): Promise<RevisionSummary[]> {
  return request<RevisionSummary[]>(`/projects/${projectId}/revisions`);
}

export function createRevision(projectId: string, instruction: string): Promise<RevisionAccepted> {
  return request<RevisionAccepted>(`/projects/${projectId}/revisions`, {
    method: "POST",
    body: JSON.stringify({ instruction, locked_ranges: [] }),
  });
}

export function createDeliveryLink(
  projectId: string,
  options: { version?: number; download?: boolean } = {},
): Promise<DeliveryLink> {
  const search = new URLSearchParams();
  if (options.version) search.set("version", String(options.version));
  if (options.download) search.set("download", "true");
  const query = search.size ? `?${search.toString()}` : "";
  return request<DeliveryLink>(`/projects/${projectId}/delivery${query}`, { method: "POST" });
}
