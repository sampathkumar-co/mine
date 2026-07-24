import { ApiError, getAccessToken } from "@/lib/api";

const API_URL = (process.env.NEXT_PUBLIC_DIRECTOR_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

export interface BillingPlan {
  key: string;
  name: string;
  description: string;
  monthly_credits: string;
  max_source_clips: number;
  max_target_duration_seconds: number;
  max_members: number;
  max_tier: number;
  checkout_available: boolean;
  current: boolean;
}

export interface WorkspaceSubscription {
  workspace_id: string;
  provider: string;
  plan_key: string;
  status: string;
  current_period_start: string | null;
  current_period_end: string | null;
  cancel_at_period_end: boolean;
  last_payment_failed_at: string | null;
  updated_at: string;
}

export interface SubscriptionOverview {
  workspace_id: string;
  plan: BillingPlan;
  subscription: WorkspaceSubscription | null;
  balance_credits: string;
  reserved_credits: string;
  available_credits: string;
  portal_available: boolean;
}

interface HostedBillingSession {
  url: string;
  expires_at: string | null;
}

async function subscriptionRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAccessToken();
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
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
      // Preserve the status-based fallback.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function listBillingPlans(): Promise<BillingPlan[]> {
  return subscriptionRequest<BillingPlan[]>("/billing/plans");
}

export function getSubscriptionOverview(workspaceId: string): Promise<SubscriptionOverview> {
  return subscriptionRequest<SubscriptionOverview>(`/workspaces/${workspaceId}/subscription`);
}

export function createSubscriptionCheckout(
  workspaceId: string,
  planKey: string,
): Promise<HostedBillingSession> {
  return subscriptionRequest<HostedBillingSession>(
    `/workspaces/${workspaceId}/subscription/checkout`,
    { method: "POST", body: JSON.stringify({ plan_key: planKey }) },
  );
}

export function createSubscriptionPortal(workspaceId: string): Promise<HostedBillingSession> {
  return subscriptionRequest<HostedBillingSession>(
    `/workspaces/${workspaceId}/subscription/portal`,
    { method: "POST" },
  );
}
