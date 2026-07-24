import {
  ApiError,
  getAccessToken,
  refreshAccessSession,
  setSessionTokens,
} from "@/lib/api";
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

async function request<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const makeRequest = () => fetch(`${API_URL}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(getAccessToken() ? { Authorization: `Bearer ${getAccessToken()}` } : {}),
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  let response = await makeRequest();
  if (response.status === 401 && retry) {
    await refreshAccessSession();
    response = await makeRequest();
  }
  if (!response.ok) {
    if (response.status === 401) setSessionTokens(null);
    throw await errorFrom(response);
  }
  return (await response.json()) as T;
}

export function resolvePrivacyDeliveryUrl(value: string): string {
  if (/^https?:\/\//i.test(value) || typeof window === "undefined") return value;
  if (value.startsWith("/api/v1")) {
    return API_URL.startsWith("http") ? new URL(value, new URL(API_URL).origin).toString() : value;
  }
  return `${API_URL}${value.startsWith("/") ? value : `/${value}`}`;
}
export function listPrivacyRequests(workspaceId: string): Promise<PrivacyRequest[]> {
  return request(`/workspaces/${workspaceId}/privacy/requests`);
}

export function createWorkspaceExport(workspaceId: string): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/exports`, { method: "POST" });
}

export function createPrivacyDelivery(workspaceId: string, requestId: string): Promise<PrivacyDelivery> {
  return request(`/workspaces/${workspaceId}/privacy/requests/${requestId}/delivery`);
}

export function scheduleWorkspaceDeletion(workspaceId: string, confirmation: string, reason: string): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/deletion`, {
    method: "POST",
    body: JSON.stringify({ confirmation, reason }),
  });
}

export function cancelWorkspaceDeletion(workspaceId: string, requestId: string): Promise<PrivacyRequest> {
  return request(`/workspaces/${workspaceId}/privacy/deletion/${requestId}`, { method: "DELETE" });
}
