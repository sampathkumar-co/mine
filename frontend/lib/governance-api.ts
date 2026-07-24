import {
  ApiError,
  getAccessToken,
  getRefreshToken,
  setSessionTokens,
} from "@/lib/api";
import type { AuthSession } from "@/lib/types";
import type { PrivacyDelivery, PrivacyRequest } from "@/lib/governance-types";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

async function errorFrom(response: Response): Promise<ApiError> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {
    // Preserve status fallback for non-JSON responses.
  }
  return new ApiError(message, response.status);
}

async function refresh(): Promise<void> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) throw new ApiError("Session has expired", 401);
  const response = await fetch(`${API_URL}/auth/refresh`, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
    cache: "no-store",
  });
  if (!response.ok) {
    setSessionTokens(null);
    throw await errorFrom(response);
  }
  setSessionTokens((await response.json()) as AuthSession);
}

async function request<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const token = getAccessToken();
  let response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (response.status === 401 && retry && getRefreshToken()) {
    await refresh();
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${getAccessToken()}`,
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      cache: "no-store",
    });
  }
  if (!response.ok) throw await errorFrom(response);
  return (await response.json()) as T;
}

export function listPrivacyRequests(workspaceId: string): Promise<PrivacyRequest[]> {
  return request(`/workspaces/${workspaceId}/privacy/requests`);
}

export function createWorkspaceExport(workspaceId: string): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/exports`, { method: "POST" });
}

export function createPrivacyDelivery(
  workspaceId: string,
  requestId: string,
): Promise<PrivacyDelivery> {
  return request(`/workspaces/${workspaceId}/privacy/requests/${requestId}/delivery`);
}

export function scheduleWorkspaceDeletion(
  workspaceId: string,
  confirmation: string,
  reason: string,
): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/deletion`, {
    method: "POST",
    body: JSON.stringify({ confirmation, reason }),
  });
}

export function cancelWorkspaceDeletion(
  workspaceId: string,
  requestId: string,
): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/deletion/${requestId}`, {
    method: "DELETE",
  });
}

export function resolvePrivacyDeliveryUrl(value: string): string {
  if (/^https?:\/\//i.test(value)) return value;
  if (typeof window === "undefined") return value;
  if (API_URL.startsWith("http")) return new URL(value, new URL(API_URL).origin).toString();
  return new URL(value, window.location.origin).toString();
}
