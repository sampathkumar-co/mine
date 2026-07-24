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

export interface User {
  id: string;
  email: string;
  display_name: string;
  created_at: string;
}

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  role: string;
  created_at: string;
}

export interface AuthSession {
  access_token: string;
  token_type: "bearer";
  expires_at: string;
  user: User;
  workspaces: Workspace[];
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

export interface ResumableUpload {
  id: string;
  project_id: string;
  asset_id: string | null;
  kind: string;
  original_filename: string;
  content_type: string;
  total_bytes: number;
  received_bytes: number;
  status: string;
  error_message: string | null;
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
