export type PrivacyRequestKind = "export" | "deletion";
export type PrivacyRequestStatus =
  | "queued"
  | "processing"
  | "ready"
  | "scheduled"
  | "completed"
  | "failed"
  | "cancelled";

export interface PrivacyRequest {
  id: string;
  workspace_id: string;
  requested_by_user_id: string | null;
  kind: PrivacyRequestKind;
  status: PrivacyRequestStatus;
  result_sha256: string | null;
  result_size_bytes: number | null;
  available_until: string | null;
  execute_after: string | null;
  completed_at: string | null;
  error_message: string | null;
  request_metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PrivacyDelivery {
  request_id: string;
  url: string;
  expires_at: string;
  sha256: string;
  size_bytes: number;
}
