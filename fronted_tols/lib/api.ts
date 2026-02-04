import type {
  PredictionResult,
  BatchPredictionResult,
  ClassInfo,
  DatasetResponse,
  DatasetListResponse,
  JobResponse,
  JobListResponse,
  ModelResponse,
  ModelListResponse,
  TrainRequest,
  HealthResponse,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  return res.json();
}

// ── Health ──────────────────────────────────────────────────────────────

export function healthCheck(): Promise<HealthResponse> {
  return request("/api/v1/health");
}

// ── Inference ───────────────────────────────────────────────────────────

export function predictImage(
  file: File,
  scoreThreshold = 0.5,
  modelId?: number,
): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("score_threshold", scoreThreshold.toString());
  if (modelId !== undefined) {
    form.append("model_id", modelId.toString());
  }
  return request("/api/v1/predict/upload", { method: "POST", body: form });
}

export function predictBatch(
  files: File[],
  scoreThreshold = 0.5,
  modelId?: number,
): Promise<BatchPredictionResult> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("score_threshold", scoreThreshold.toString());
  if (modelId !== undefined) {
    form.append("model_id", modelId.toString());
  }
  return request("/api/v1/predict/batch", { method: "POST", body: form });
}

export function getClasses(): Promise<{ classes: ClassInfo[] }> {
  return request("/api/v1/classes");
}

// ── Datasets ────────────────────────────────────────────────────────────

export function getDatasets(status = "active"): Promise<DatasetListResponse> {
  return request(`/api/v1/datasets?status=${status}`);
}

export function uploadDataset(
  name: string,
  description: string,
  images: File[],
  labels?: File[],
  annotationFile?: File,
): Promise<DatasetResponse> {
  const form = new FormData();
  form.append("name", name);
  form.append("description", description);
  images.forEach((f) => form.append("images", f));
  if (labels && labels.length > 0) {
    labels.forEach((f) => form.append("labels", f));
  }
  if (annotationFile) {
    form.append("annotation_file", annotationFile);
  }
  return request("/api/v1/datasets", { method: "POST", body: form });
}

// ── Training Jobs ───────────────────────────────────────────────────────

export function getJobs(status?: string): Promise<JobListResponse> {
  const qs = status ? `?status=${status}` : "";
  return request(`/api/v1/jobs${qs}`);
}

export function createTrainingJob(config: TrainRequest): Promise<JobResponse> {
  return request("/api/v1/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export function cancelJob(jobId: number): Promise<JobResponse> {
  return request(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

// ── Models ──────────────────────────────────────────────────────────────

export function getModels(environment?: string): Promise<ModelListResponse> {
  const qs = environment ? `?environment=${environment}` : "";
  return request(`/api/v1/models${qs}`);
}

export function promoteModel(modelId: number): Promise<ModelResponse> {
  return request(`/api/v1/models/${modelId}/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ environment: "production" }),
  });
}
