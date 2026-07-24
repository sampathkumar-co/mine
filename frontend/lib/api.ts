import type {
  AuditEvent,
  AuthSession,
  BillingAccount,
  BillingEntry,
  CameraMode,
  DeliveryLink,
  DirectorCamera,
  MultipartPart,
  MultipartPartTarget,
  MultipartUpload,
  Project,
  ProjectAccepted,
  RevisionAccepted,
  RevisionSummary,
  Workspace,
  WorkspaceInvitation,
  WorkspaceMember,
  WorkspaceProject,
  WorkspaceRole,
} from "@/lib/types";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");
const ACCESS_TOKEN_KEY = "director-os-access-token";
const REFRESH_TOKEN_KEY = "director-os-refresh-token";
let refreshInFlight: Promise<AuthSession> | null = null;

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
  return window.sessionStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(REFRESH_TOKEN_KEY);
}

function dispatchAuthChange(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event("director-auth-changed"));
}

export function setAccessToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.sessionStorage.setItem(ACCESS_TOKEN_KEY, token);
  else window.sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  dispatchAuthChange();
}

export function setSessionTokens(session: AuthSession | null): void {
  if (typeof window === "undefined") return;
  if (!session) {
    window.sessionStorage.removeItem(ACCESS_TOKEN_KEY);
    window.sessionStorage.removeItem(REFRESH_TOKEN_KEY);
  } else {
    window.sessionStorage.setItem(ACCESS_TOKEN_KEY, session.access_token);
    if (session.refresh_token) window.sessionStorage.setItem(REFRESH_TOKEN_KEY, session.refresh_token);
  }
  dispatchAuthChange();
}

function authHeaders(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function apiOrigin(): string {
  if (typeof window === "undefined") return "";
  return new URL(API_URL, window.location.origin).origin;
}

function resolveTargetUrl(value: string): string {
  if (/^https?:\/\//i.test(value)) return value;
  if (value.startsWith("/api/v1")) {
    return API_URL.startsWith("http") ? new URL(value, apiOrigin()).toString() : value;
  }
  return `${API_URL}${value.startsWith("/") ? value : `/${value}`}`;
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {
    // Keep the status-based message when the response is not JSON.
  }
  return new ApiError(message, response.status);
}

function structuredBody(body: BodyInit | null | undefined): boolean {
  return Boolean(body && !(body instanceof FormData) && !(body instanceof Blob));
}

async function fetchApi(path: string, init?: RequestInit, withAuth = true): Promise<Response> {
  return fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(withAuth ? authHeaders() : {}),
      ...(structuredBody(init?.body) ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
}

async function refreshAccessSession(): Promise<AuthSession> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) throw new ApiError("Session has expired", 401);
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      const response = await fetchApi(
        "/auth/refresh",
        { method: "POST", body: JSON.stringify({ refresh_token: refreshToken }) },
        false,
      );
      if (!response.ok) {
        setSessionTokens(null);
        throw await responseError(response);
      }
      const session = (await response.json()) as AuthSession;
      setSessionTokens(session);
      return session;
    })().finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

async function request<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  let response = await fetchApi(path, init);
  if (response.status === 401 && retry && getRefreshToken() && path !== "/auth/refresh") {
    await refreshAccessSession();
    response = await fetchApi(path, init);
  }
  if (!response.ok) {
    if (response.status === 401) setSessionTokens(null);
    throw await responseError(response);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function upgradeBootstrapSession(): Promise<AuthSession> {
  const session = await request<AuthSession>("/auth/session", { method: "POST" }, false);
  setSessionTokens(session);
  return session;
}

export async function register(input: {
  email: string;
  password: string;
  displayName: string;
  workspaceName: string;
}): Promise<AuthSession> {
  const bootstrap = await fetchApi(
    "/auth/register",
    {
      method: "POST",
      body: JSON.stringify({
        email: input.email,
        password: input.password,
        display_name: input.displayName,
        workspace_name: input.workspaceName,
      }),
    },
    false,
  );
  if (!bootstrap.ok) throw await responseError(bootstrap);
  const session = (await bootstrap.json()) as AuthSession;
  setAccessToken(session.access_token);
  return upgradeBootstrapSession();
}

export async function login(email: string, password: string): Promise<AuthSession> {
  const response = await fetchApi(
    "/auth/login",
    { method: "POST", body: JSON.stringify({ email, password }) },
    false,
  );
  if (!response.ok) throw await responseError(response);
  const session = (await response.json()) as AuthSession;
  setAccessToken(session.access_token);
  return upgradeBootstrapSession();
}

export async function getSession(): Promise<AuthSession> {
  const session = await request<AuthSession>("/auth/me");
  return {
    ...session,
    access_token: getAccessToken() ?? session.access_token,
    refresh_token: getRefreshToken(),
  };
}

export async function logout(): Promise<void> {
  try {
    await request("/auth/logout", {
      method: "POST",
      body: JSON.stringify({ refresh_token: getRefreshToken() }),
    });
  } finally {
    setSessionTokens(null);
  }
}

export function logoutAll(): Promise<{ message: string }> {
  return request("/auth/logout-all", { method: "POST" });
}

export function requestEmailVerification(): Promise<{ message: string }> {
  return request("/auth/email-verification/request", { method: "POST" });
}

export function confirmEmailVerification(token: string): Promise<{ message: string }> {
  return request("/auth/email-verification/confirm", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export function requestPasswordReset(email: string): Promise<{ message: string }> {
  return request("/auth/password-reset/request", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function confirmPasswordReset(token: string, newPassword: string): Promise<{ message: string }> {
  return request("/auth/password-reset/confirm", {
    method: "POST",
    body: JSON.stringify({ token, new_password: newPassword }),
  });
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

export function listWorkspaceMembers(workspaceId: string): Promise<WorkspaceMember[]> {
  return request<WorkspaceMember[]>(`/workspaces/${workspaceId}/members`);
}

export function updateWorkspaceMember(
  workspaceId: string,
  membershipId: string,
  role: WorkspaceRole,
): Promise<WorkspaceMember> {
  return request<WorkspaceMember>(`/workspaces/${workspaceId}/members/${membershipId}`, {
    method: "PATCH",
    body: JSON.stringify({ role }),
  });
}

export function removeWorkspaceMember(workspaceId: string, membershipId: string): Promise<{ message: string }> {
  return request(`/workspaces/${workspaceId}/members/${membershipId}`, { method: "DELETE" });
}

export function listWorkspaceInvitations(workspaceId: string): Promise<WorkspaceInvitation[]> {
  return request<WorkspaceInvitation[]>(`/workspaces/${workspaceId}/invitations`);
}

export function createWorkspaceInvitation(
  workspaceId: string,
  email: string,
  role: WorkspaceRole,
): Promise<WorkspaceInvitation> {
  return request<WorkspaceInvitation>(`/workspaces/${workspaceId}/invitations`, {
    method: "POST",
    body: JSON.stringify({ email, role }),
  });
}

export function revokeWorkspaceInvitation(workspaceId: string, invitationId: string): Promise<{ message: string }> {
  return request(`/workspaces/${workspaceId}/invitations/${invitationId}`, { method: "DELETE" });
}

export function acceptWorkspaceInvitation(token: string): Promise<{ workspace_id: string }> {
  return request("/invitations/accept", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export function listAuditEvents(workspaceId: string): Promise<AuditEvent[]> {
  return request<AuditEvent[]>(`/workspaces/${workspaceId}/audit-events`);
}

export function getBillingAccount(workspaceId: string): Promise<BillingAccount> {
  return request<BillingAccount>(`/workspaces/${workspaceId}/billing`);
}

export function listBillingEntries(workspaceId: string): Promise<BillingEntry[]> {
  return request<BillingEntry[]>(`/workspaces/${workspaceId}/billing/entries`);
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
  const upload = await request<MultipartUpload>(`/projects/${projectId}/multipart-uploads`, {
    method: "POST",
    body: JSON.stringify({
      kind,
      original_filename: file.name,
      content_type: file.type || "video/mp4",
      total_bytes: file.size,
    }),
  });
  const partCount = Math.ceil(file.size / upload.part_size);
  const completedParts: MultipartPart[] = [];
  for (let partNumber = 1; partNumber <= partCount; partNumber += 1) {
    const start = (partNumber - 1) * upload.part_size;
    const end = Math.min(file.size, start + upload.part_size);
    const chunk = file.slice(start, end, "application/octet-stream");
    const target = await request<MultipartPartTarget>(
      `/multipart-uploads/${upload.id}/parts/${partNumber}/target`,
      { method: "POST" },
    );
    const targetUrl = resolveTargetUrl(target.url);
    const localTarget = targetUrl.startsWith(apiOrigin()) || targetUrl.startsWith("/api/v1");
    const response = await fetch(targetUrl, {
      method: target.method,
      headers: {
        ...target.headers,
        ...(localTarget ? authHeaders() : {}),
      },
      body: chunk,
    });
    if (!response.ok) throw await responseError(response);
    let part: MultipartPart;
    if (localTarget) {
      part = (await response.json()) as MultipartPart;
    } else {
      const etag = response.headers.get("etag")?.replaceAll('"', "");
      if (!etag) throw new ApiError("Object storage did not return a part ETag", 502);
      part = { part_number: partNumber, etag, size_bytes: chunk.size };
      await request(`/multipart-uploads/${upload.id}/parts`, {
        method: "POST",
        body: JSON.stringify(part),
      });
    }
    completedParts.push(part);
    onProgress?.(partNumber / partCount);
  }
  await request(`/multipart-uploads/${upload.id}/complete`, {
    method: "POST",
    body: JSON.stringify({ parts: completedParts }),
  });
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
  let response = await fetch(
    `${API_URL}/projects/${projectId}/director-camera/missions/${missionId}/ghost-frame`,
    { headers: authHeaders(), cache: "no-store" },
  );
  if (response.status === 401 && getRefreshToken()) {
    await refreshAccessSession();
    response = await fetch(
      `${API_URL}/projects/${projectId}/director-camera/missions/${missionId}/ghost-frame`,
      { headers: authHeaders(), cache: "no-store" },
    );
  }
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

export async function createDeliveryLink(
  projectId: string,
  options: { version?: number; download?: boolean } = {},
): Promise<DeliveryLink> {
  const search = new URLSearchParams();
  if (options.version) search.set("version", String(options.version));
  if (options.download) search.set("download", "true");
  const query = search.toString() ? `?${search.toString()}` : "";
  const link = await request<DeliveryLink>(`/projects/${projectId}/delivery${query}`, { method: "POST" });
  if (!link.url.startsWith("http")) link.url = new URL(link.url, apiOrigin()).toString();
  return link;
}
