export type ProjectStatus =
  | "created"
  | "uploading"
  | "ready_to_queue"
  | "queued"
  | "analyzing"
  | "needs_pickups"
  | "planning"
  | "rendering"
  | "quality_check"
  | "ready"
  | "failed";

export type CameraMode = "off" | "advisory" | "required";
export type WorkspaceRole = "owner" | "admin" | "editor" | "viewer";

export interface User {
  id: string;
  email: string;
  display_name: string;
  email_verified: boolean;
  created_at: string;
}

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  role: WorkspaceRole;
  created_at: string;
}

export interface AuthSession {
  access_token: string;
  refresh_token: string | null;
  token_type: "bearer";
  expires_at: string;
  refresh_expires_at: string | null;
  session_id: string | null;
  user: User;
  workspaces: Workspace[];
}

export interface WorkspaceMember {
  id: string;
  user_id: string;
  email: string;
  display_name: string;
  role: WorkspaceRole;
  created_at: string;
}

export interface WorkspaceInvitation {
  id: string;
  workspace_id: string;
  email: string;
  role: WorkspaceRole;
  expires_at: string;
  accepted_at: string | null;
  revoked_at: string | null;
  created_at: string;
}

export interface AuditEvent {
  id: string;
  workspace_id: string | null;
  actor_user_id: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  request_id: string | null;
  ip_address: string | null;
  user_agent: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface BillingAccount {
  workspace_id: string;
  plan: string;
  balance_credits: string;
  reserved_credits: string;
  available_credits: string;
  updated_at: string;
}

export interface BillingEntry {
  id: string;
  workspace_id: string;
  project_id: string | null;
  actor_user_id: string | null;
  kind: string;
  amount_credits: string;
  idempotency_key: string;
  description: string;
  entry_metadata: Record<string, unknown>;
  created_at: string;
}

export interface MultipartUpload {
  id: string;
  project_id: string;
  asset_id: string | null;
  provider: "local" | "s3" | string;
  object_key: string;
  kind: string;
  original_filename: string;
  content_type: string;
  total_bytes: number;
  part_size: number;
  status: string;
  error_message: string | null;
  expires_at: string;
  created_at: string;
  updated_at: string;
}

export interface MultipartUploadDetail extends MultipartUpload {
  parts: MultipartPart[];
}

export interface MultipartPartTarget {
  upload_id: string;
  part_number: number;
  expected_size: number;
  method: string;
  url: string;
  headers: Record<string, string>;
}

export interface MultipartPart {
  part_number: number;
  etag: string;
  size_bytes: number;
}

export interface DirectorContract {
  objective: string;
  target_audience?: string | null;
  target_platform: string;
  target_duration_seconds: number;
  tier: number;
  instructions?: string | null;
  must_include: string[];
  must_avoid: string[];
  reference_rules: Record<string, string>;
  brand_rules: Record<string, unknown>;
  creative_freedom: number;
  director_profile_key: string;
  use_director_memory: boolean;
  director_camera_mode: CameraMode;
  production_readiness_threshold: number;
}

export interface ProjectAsset {
  id: string;
  kind: string;
  original_filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export interface Project {
  id: string;
  user_id: string;
  workspace_id: string | null;
  status: ProjectStatus;
  contract: DirectorContract;
  task_id: string | null;
  output_available: boolean;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  assets: ProjectAsset[];
}

export interface WorkspaceProject {
  id: string;
  workspace_id: string;
  status: ProjectStatus;
  objective: string;
  target_platform: string;
  target_duration_seconds: number;
  output_available: boolean;
  asset_count: number;
  created_at: string;
  updated_at: string;
}

export interface DeliveryLink {
  project_id: string;
  revision_version: number | null;
  url: string;
  expires_at: string;
  download: boolean;
}

export interface RevisionSummary {
  version: number;
  base_version: number | null;
  instruction: string | null;
  status: string;
  task_id: string | null;
  is_active: boolean;
  output_available: boolean;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface RevisionAccepted {
  project_id: string;
  version: number;
  base_version: number;
  status: string;
  task_id: string;
  message: string;
}

export interface CameraDimension {
  name?: string;
  score?: number;
  weight?: number;
  blocking?: boolean;
  findings?: string[];
}

export interface CameraReport {
  dimensions?: Record<string, CameraDimension> | CameraDimension[];
  findings?: string[];
  recommendations?: string[];
  [key: string]: unknown;
}

export interface PickupMission {
  id: string;
  mission_type: string;
  priority: string;
  title: string;
  reason: string;
  status: string;
  specification: Record<string, unknown>;
  target_terms: string[];
  submitted_asset_id: string | null;
  accepted_asset_id: string | null;
  validation: Record<string, unknown>;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DirectorCamera {
  project_id: string;
  project_status: ProjectStatus;
  audit_id: string | null;
  audit_version: number | null;
  mode: CameraMode;
  readiness_score: number | null;
  threshold: number | null;
  ready: boolean | null;
  report: CameraReport;
  missions: PickupMission[];
}

export interface ProjectAccepted {
  project_id: string;
  status: ProjectStatus;
  task_id: string;
  message: string;
}
