import { getAccessToken } from "@/lib/api";
import type { User, Workspace } from "@/lib/types";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

export interface AccountContext {
  user: User;
  workspaces: Workspace[];
}

export async function getAccountContext(): Promise<AccountContext> {
  const token = getAccessToken();
  const response = await fetch(`${API_URL}/auth/account`, {
    headers: {
      Accept: "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `Account request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Preserve the status-based fallback.
    }
    throw new Error(message);
  }
  return (await response.json()) as AccountContext;
}
