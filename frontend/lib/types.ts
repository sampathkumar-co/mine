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
  status: ProjectStatus;
  contract: DirectorContract;
  task_id: string | null;
  output_available: boolean;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  assets: ProjectAsset[];
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
