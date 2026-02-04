export interface Detection {
  class_id: number;
  class_name: string;
  score: number;
  bbox: number[];
  mask_rle: string | null;
}

export interface PredictionResult {
  detections: Detection[];
  inference_time_ms: number;
  model_version: string;
  image_size: number[];
  annotated_image: string;
}

export interface BatchItemResult {
  filename: string;
  detections: Detection[];
  inference_time_ms: number;
  annotated_image: string;
}

export interface BatchPredictionResult {
  results: BatchItemResult[];
  total_time_ms: number;
  model_version: string;
}

export interface ClassInfo {
  id: number;
  name: string;
  color: string;
}

export interface DatasetResponse {
  id: number;
  name: string;
  description: string;
  image_dir: string | null;
  label_dir: string | null;
  num_images: number;
  num_annotations: number;
  status: string;
  created_at: string;
}

export interface DatasetListResponse {
  datasets: DatasetResponse[];
  total: number;
}

export interface JobResponse {
  id: number;
  job_type: string;
  dataset_id: number | null;
  model_id: number | null;
  status: string;
  pid: number | null;
  config: Record<string, unknown>;
  mlflow_run_id: string | null;
  output_model_path: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobListResponse {
  jobs: JobResponse[];
  total: number;
}

export interface ModelResponse {
  id: number;
  name: string;
  version: string;
  file_path: string;
  environment: string;
  training_job_id: number | null;
  mlflow_run_id: string | null;
  metrics: Record<string, number>;
  created_at: string;
  promoted_at: string | null;
}

export interface ModelListResponse {
  models: ModelResponse[];
  total: number;
}

export interface TrainRequest {
  job_type: string;
  dataset_id: number;
  base_model_id: number | null;
  config: {
    num_epochs: number;
    learning_rate: number;
    batch_size: number;
    freeze_epochs: number;
    device: string;
  };
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  device: string;
}

export interface AnnotationBox {
  class_id: number;
  class_name: string;
  bbox: number[]; // [x1, y1, x2, y2] in pixels
}

export interface ImageAnnotation {
  filename: string;
  width: number;
  height: number;
  boxes: AnnotationBox[];
}

export const CLASS_COLORS: Record<string, string> = {
  "Adjustable spanner": "#e6194b",
  "Backsaw": "#3cb44b",
  "Calipers": "#ffe119",
  "Cutting pliers": "#4363d8",
  "Drill": "#f58231",
  "Gas wrench": "#911eb4",
  "Gun": "#42d4f4",
  "Hammer": "#f032e6",
  "Hand": "#bfef45",
  "Handsaw": "#fabed4",
  "Needle-nose pliers": "#469990",
  "Pliers": "#dcbeff",
  "Ratchet": "#9a6324",
  "Screwdriver": "#800000",
  "Tape measure": "#aaffc3",
  "Utility knife": "#808000",
  "Wrench": "#000075",
};
