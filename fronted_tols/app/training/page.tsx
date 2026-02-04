"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  getDatasets,
  uploadDataset,
  getJobs,
  createTrainingJob,
  cancelJob,
  getModels,
  promoteModel,
  getClasses,
} from "@/lib/api";
import type {
  DatasetResponse,
  JobResponse,
  ModelResponse,
  TrainRequest,
  ClassInfo,
  ImageAnnotation,
} from "@/lib/types";
import { ModelSelector } from "@/components/model-selector";
import { BboxAnnotator } from "@/components/bbox-annotator";
import {
  Upload,
  Play,
  X,
  ArrowUpCircle,
  Loader2,
  Plus,
  Database,
  Cpu,
  Layers,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  PenTool,
} from "lucide-react";

type Tab = "datasets" | "jobs" | "models";

export default function TrainingPage() {
  const [activeTab, setActiveTab] = useState<Tab>("datasets");

  const tabs: { key: Tab; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { key: "datasets", label: "Datasets", icon: Database },
    { key: "jobs", label: "Jobs", icon: Cpu },
    { key: "models", label: "Models", icon: Layers },
  ];

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold tracking-tight">
        Training Management
      </h1>
      <p className="text-text-secondary text-sm mt-1">
        Manage datasets, launch training jobs, and promote models
      </p>

      {/* Tab bar */}
      <div className="flex border-b border-border mt-8">
        {tabs.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === key
                ? "text-text border-accent"
                : "text-text-muted hover:text-text-secondary border-transparent"
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-6">
        {activeTab === "datasets" && <DatasetsTab />}
        {activeTab === "jobs" && <JobsTab />}
        {activeTab === "models" && <ModelsTab />}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   DATASETS TAB
   ═══════════════════════════════════════════════════════════════════════ */

function DatasetsTab() {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    getDatasets()
      .then((r) => setDatasets(r.datasets))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-text-secondary">
          {datasets.length} dataset{datasets.length !== 1 && "s"}
        </p>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2 text-sm font-medium transition-colors"
        >
          {showForm ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <Plus className="h-4 w-4" />
          )}
          {showForm ? "Cancel" : "Upload Dataset"}
        </button>
      </div>

      {showForm && (
        <DatasetUploadForm
          onSuccess={() => {
            setShowForm(false);
            load();
          }}
        />
      )}

      {loading ? (
        <LoadingState />
      ) : datasets.length === 0 ? (
        <EmptyState message="No datasets yet" />
      ) : (
        <div className="border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-text-muted text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Name</th>
                <th className="text-left px-4 py-3 font-medium">Images</th>
                <th className="text-left px-4 py-3 font-medium">
                  Annotations
                </th>
                <th className="text-left px-4 py-3 font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((ds) => (
                <tr
                  key={ds.id}
                  className="border-b border-border last:border-0 hover:bg-surface-alt transition-colors"
                >
                  <td className="px-4 py-3 font-medium">{ds.name}</td>
                  <td className="px-4 py-3 font-mono text-text-secondary">
                    {ds.num_images}
                  </td>
                  <td className="px-4 py-3 font-mono text-text-secondary">
                    {ds.num_annotations}
                  </td>
                  <td className="px-4 py-3 text-text-muted">
                    {formatDate(ds.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

type UploadMode = "annotate" | "files";

function DatasetUploadForm({ onSuccess }: { onSuccess: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [uploadMode, setUploadMode] = useState<UploadMode>("annotate");
  const [images, setImages] = useState<File[]>([]);
  const [labelFiles, setLabelFiles] = useState<File[]>([]);
  const [annotations, setAnnotations] = useState<ImageAnnotation[]>([]);
  const [classes, setClasses] = useState<ClassInfo[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getClasses()
      .then((r) => setClasses(r.classes))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || images.length === 0) return;

    setSubmitting(true);
    setError(null);
    try {
      if (uploadMode === "annotate" && annotations.length > 0) {
        // Convert annotations to YOLO label files
        const labelBlobs: File[] = [];
        for (const ann of annotations) {
          const lines = ann.boxes.map((box) => {
            const cx = ((box.bbox[0] + box.bbox[2]) / 2) / ann.width;
            const cy = ((box.bbox[1] + box.bbox[3]) / 2) / ann.height;
            const w = (box.bbox[2] - box.bbox[0]) / ann.width;
            const h = (box.bbox[3] - box.bbox[1]) / ann.height;
            return `${box.class_id} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
          });
          const stem = ann.filename.replace(/\.[^.]+$/, "");
          const blob = new Blob([lines.join("\n") + "\n"], {
            type: "text/plain",
          });
          labelBlobs.push(new File([blob], `${stem}.txt`));
        }
        await uploadDataset(name, description, images, labelBlobs);
      } else if (uploadMode === "files" && labelFiles.length > 0) {
        await uploadDataset(name, description, images, labelFiles);
      } else {
        await uploadDataset(name, description, images);
      }
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  };

  const totalAnnotations = annotations.reduce(
    (sum, a) => sum + a.boxes.length,
    0,
  );

  return (
    <form
      onSubmit={handleSubmit}
      className="border border-border bg-surface p-5 space-y-4"
    >
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Name *
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Description
          </label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text outline-none focus:border-accent"
          />
        </div>
      </div>

      {/* Upload mode toggle */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setUploadMode("annotate")}
          className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium border transition-colors ${
            uploadMode === "annotate"
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-secondary hover:bg-surface-alt"
          }`}
        >
          <PenTool className="h-3 w-3" />
          Draw Annotations
        </button>
        <button
          type="button"
          onClick={() => setUploadMode("files")}
          className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium border transition-colors ${
            uploadMode === "files"
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-secondary hover:bg-surface-alt"
          }`}
        >
          <Upload className="h-3 w-3" />
          Upload Label Files
        </button>
      </div>

      {/* Images upload */}
      <div>
        <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
          Images *
        </label>
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => setImages(Array.from(e.target.files || []))}
          required
          className="w-full text-sm text-text-secondary file:mr-3 file:px-3 file:py-1.5 file:text-xs file:font-medium file:bg-surface-alt file:text-text-secondary file:border file:border-border file:cursor-pointer"
        />
        {images.length > 0 && (
          <p className="text-xs text-text-muted mt-1">
            {images.length} file{images.length !== 1 && "s"} selected
          </p>
        )}
      </div>

      {/* Annotation mode: bbox annotator */}
      {uploadMode === "annotate" && images.length > 0 && (
        <div>
          <p className="text-xs text-text-muted uppercase tracking-wide mb-2">
            Annotate Images
          </p>
          <BboxAnnotator
            images={images}
            classNames={classes.map((c) => ({ id: c.id, name: c.name }))}
            onAnnotationsChange={setAnnotations}
          />
        </div>
      )}

      {/* File mode: YOLO label files */}
      {uploadMode === "files" && (
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            YOLO Label Files (.txt)
          </label>
          <input
            type="file"
            accept=".txt"
            multiple
            onChange={(e) => setLabelFiles(Array.from(e.target.files || []))}
            className="w-full text-sm text-text-secondary file:mr-3 file:px-3 file:py-1.5 file:text-xs file:font-medium file:bg-surface-alt file:text-text-secondary file:border file:border-border file:cursor-pointer"
          />
          {labelFiles.length > 0 && (
            <p className="text-xs text-text-muted mt-1">
              {labelFiles.length} label file{labelFiles.length !== 1 && "s"}{" "}
              selected
            </p>
          )}
        </div>
      )}

      {error && (
        <div className="border border-danger/30 bg-danger/10 text-danger p-3 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={submitting || !name || images.length === 0}
        className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
      >
        {submitting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Upload className="h-4 w-4" />
        )}
        {submitting
          ? "Uploading..."
          : uploadMode === "annotate" && totalAnnotations > 0
            ? `Upload Dataset (${totalAnnotations} annotations)`
            : "Upload Dataset"}
      </button>
    </form>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   JOBS TAB
   ═══════════════════════════════════════════════════════════════════════ */

function JobsTab() {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([getJobs(), getDatasets()])
      .then(([j, d]) => {
        setJobs(j.jobs);
        setDatasets(d.datasets);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll for running jobs — use derived boolean to avoid interval churn
  const hasRunning = jobs.some((j) => j.status === "running" || j.status === "pending");
  useEffect(() => {
    if (!hasRunning) return;
    pollRef.current = setInterval(() => {
      getJobs()
        .then((r) => setJobs(r.jobs))
        .catch(() => {});
    }, 5000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [hasRunning]);

  const handleCancel = async (jobId: number) => {
    try {
      await cancelJob(jobId);
      load();
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <p className="text-sm text-text-secondary">
            {jobs.length} job{jobs.length !== 1 && "s"}
          </p>
          <button
            onClick={load}
            className="text-text-muted hover:text-text-secondary transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2 text-sm font-medium transition-colors"
        >
          {showForm ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <Plus className="h-4 w-4" />
          )}
          {showForm ? "Cancel" : "New Training"}
        </button>
      </div>

      {showForm && (
        <TrainForm
          datasets={datasets}
          onSuccess={() => {
            setShowForm(false);
            load();
          }}
        />
      )}

      {loading ? (
        <LoadingState />
      ) : jobs.length === 0 ? (
        <EmptyState message="No training jobs yet" />
      ) : (
        <div className="border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-text-muted text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">ID</th>
                <th className="text-left px-4 py-3 font-medium">Type</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="text-left px-4 py-3 font-medium">Dataset</th>
                <th className="text-left px-4 py-3 font-medium">Created</th>
                <th className="text-left px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr
                  key={job.id}
                  className="border-b border-border last:border-0 hover:bg-surface-alt transition-colors"
                >
                  <td className="px-4 py-3 font-mono text-text-secondary">
                    {job.id}
                  </td>
                  <td className="px-4 py-3 text-text-secondary">
                    {job.job_type}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-3 text-text-secondary">
                    {job.dataset_id ?? "\u2014"}
                  </td>
                  <td className="px-4 py-3 text-text-muted">
                    {formatDate(job.created_at)}
                  </td>
                  <td className="px-4 py-3">
                    {job.status === "running" && (
                      <button
                        onClick={() => handleCancel(job.id)}
                        className="flex items-center gap-1 text-xs text-danger hover:text-danger/80 transition-colors"
                      >
                        <X className="h-3 w-3" />
                        Cancel
                      </button>
                    )}
                    {job.error_message && (
                      <span
                        className="text-xs text-danger cursor-help"
                        title={job.error_message}
                      >
                        Error details
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function TrainForm({
  datasets,
  onSuccess,
}: {
  datasets: DatasetResponse[];
  onSuccess: () => void;
}) {
  const [datasetId, setDatasetId] = useState<number>(datasets[0]?.id ?? 0);
  const [baseModelId, setBaseModelId] = useState<number | undefined>(undefined);
  const [epochs, setEpochs] = useState(15);
  const [lr, setLr] = useState(0.005);
  const [batchSize, setBatchSize] = useState(4);
  const [freezeEpochs, setFreezeEpochs] = useState(3);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!datasetId) return;
    setSubmitting(true);
    setError(null);
    try {
      const req: TrainRequest = {
        job_type: "fine_tune",
        dataset_id: datasetId,
        base_model_id: baseModelId ?? null,
        config: {
          num_epochs: epochs,
          learning_rate: lr,
          batch_size: batchSize,
          freeze_epochs: freezeEpochs,
          device: "auto",
        },
      };
      await createTrainingJob(req);
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="border border-border bg-surface p-5 space-y-4"
    >
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Dataset *
          </label>
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(Number(e.target.value))}
            required
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text outline-none focus:border-accent"
          >
            <option value={0} disabled>
              Select a dataset
            </option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name} ({ds.num_images} images)
              </option>
            ))}
          </select>
        </div>
        <ModelSelector
          value={baseModelId}
          onChange={setBaseModelId}
          label="Base Model"
        />
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Epochs
          </label>
          <input
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text font-mono outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Learning Rate
          </label>
          <input
            type="number"
            min={0.0001}
            step={0.001}
            value={lr}
            onChange={(e) => setLr(Number(e.target.value))}
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text font-mono outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Batch Size
          </label>
          <input
            type="number"
            min={1}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text font-mono outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
            Freeze Epochs
          </label>
          <input
            type="number"
            min={0}
            value={freezeEpochs}
            onChange={(e) => setFreezeEpochs(Number(e.target.value))}
            className="w-full bg-background border border-border px-3 py-2 text-sm text-text font-mono outline-none focus:border-accent"
          />
        </div>
      </div>

      {error && (
        <div className="border border-danger/30 bg-danger/10 text-danger p-3 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={submitting || !datasetId}
        className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
      >
        {submitting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        {submitting ? "Starting..." : "Start Training"}
      </button>
    </form>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   MODELS TAB
   ═══════════════════════════════════════════════════════════════════════ */

function ModelsTab() {
  const [models, setModels] = useState<ModelResponse[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    getModels()
      .then((r) => setModels(r.models))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handlePromote = async (id: number) => {
    try {
      await promoteModel(id);
      load();
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <p className="text-sm text-text-secondary">
          {models.length} model{models.length !== 1 && "s"}
        </p>
        <button
          onClick={load}
          className="text-text-muted hover:text-text-secondary transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      {loading ? (
        <LoadingState />
      ) : models.length === 0 ? (
        <EmptyState message="No models registered yet" />
      ) : (
        <div className="border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-text-muted text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Name</th>
                <th className="text-left px-4 py-3 font-medium">Version</th>
                <th className="text-left px-4 py-3 font-medium">
                  Environment
                </th>
                <th className="text-left px-4 py-3 font-medium">Metrics</th>
                <th className="text-left px-4 py-3 font-medium">Created</th>
                <th className="text-left px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.id}
                  className="border-b border-border last:border-0 hover:bg-surface-alt transition-colors"
                >
                  <td className="px-4 py-3 font-medium">{m.name}</td>
                  <td className="px-4 py-3 font-mono text-text-secondary">
                    {m.version}
                  </td>
                  <td className="px-4 py-3">
                    <EnvironmentBadge env={m.environment} />
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-text-secondary">
                    {Object.entries(m.metrics)
                      .map(([k, v]) => `${k}: ${(v as number).toFixed(4)}`)
                      .join(", ") || "\u2014"}
                  </td>
                  <td className="px-4 py-3 text-text-muted">
                    {formatDate(m.created_at)}
                  </td>
                  <td className="px-4 py-3">
                    {m.environment === "staging" && (
                      <button
                        onClick={() => handlePromote(m.id)}
                        className="flex items-center gap-1 text-xs text-accent hover:text-accent-hover transition-colors"
                      >
                        <ArrowUpCircle className="h-3.5 w-3.5" />
                        Promote
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SHARED COMPONENTS
   ═══════════════════════════════════════════════════════════════════════ */

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: "bg-warning/15 text-warning",
    running: "bg-accent/15 text-accent",
    completed: "bg-success/15 text-success",
    failed: "bg-danger/15 text-danger",
    cancelled: "bg-text-muted/15 text-text-muted",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium ${styles[status] || styles.cancelled}`}
    >
      {status === "running" && (
        <span className="w-1.5 h-1.5 bg-accent animate-pulse" />
      )}
      {status}
    </span>
  );
}

function EnvironmentBadge({ env }: { env: string }) {
  const styles: Record<string, string> = {
    production: "bg-success/15 text-success",
    staging: "bg-warning/15 text-warning",
    archived: "bg-text-muted/15 text-text-muted",
  };

  return (
    <span
      className={`inline-block px-2.5 py-1 text-xs font-medium ${styles[env] || styles.archived}`}
    >
      {env}
    </span>
  );
}

function LoadingState() {
  return (
    <div className="border border-border bg-surface p-8 text-center">
      <Loader2 className="h-5 w-5 animate-spin text-text-muted mx-auto" />
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="border border-border bg-surface p-8 text-center">
      <p className="text-text-muted text-sm">{message}</p>
    </div>
  );
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
