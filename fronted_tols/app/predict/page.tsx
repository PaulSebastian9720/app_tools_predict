"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { ImageDropzone } from "@/components/image-dropzone";
import { BatchDropzone } from "@/components/batch-dropzone";
import { ModelSelector } from "@/components/model-selector";
import { predictImage, predictBatch } from "@/lib/api";
import { CLASS_COLORS } from "@/lib/types";
import type {
  PredictionResult,
  BatchPredictionResult,
  BatchItemResult,
  Detection,
} from "@/lib/types";
import {
  Loader2,
  Trash2,
  Zap,
  ImageIcon,
  LayoutGrid,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

type Mode = "single" | "batch";

export default function PredictPage() {
  const [mode, setMode] = useState<Mode>("single");

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold tracking-tight">Image Prediction</h1>
      <p className="text-text-secondary text-sm mt-1">
        Upload images to detect tools using YOLOv8
      </p>

      {/* Mode toggle */}
      <div className="flex gap-2 mt-6">
        <button
          onClick={() => setMode("single")}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border transition-colors ${
            mode === "single"
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-secondary hover:bg-surface-alt"
          }`}
        >
          <ImageIcon className="h-4 w-4" />
          Single Image
        </button>
        <button
          onClick={() => setMode("batch")}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border transition-colors ${
            mode === "batch"
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-secondary hover:bg-surface-alt"
          }`}
        >
          <LayoutGrid className="h-4 w-4" />
          Batch Images
        </button>
      </div>

      {mode === "single" ? <SingleMode /> : <BatchMode />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SINGLE IMAGE MODE
   ═══════════════════════════════════════════════════════════════════════ */

function SingleMode() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [modelId, setModelId] = useState<number | undefined>(undefined);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  const handleFileSelect = useCallback((f: File) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = URL.createObjectURL(f);
    previewRef.current = url;
    setFile(f);
    setPreview(url);
    setResult(null);
    setError(null);
  }, []);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file, threshold, modelId);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const displayImage = result
    ? `data:image/png;base64,${result.annotated_image}`
    : preview;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mt-6">
      {/* Image Area */}
      <div className="lg:col-span-3 space-y-4">
        {!preview ? (
          <>
            <ModelSelector value={modelId} onChange={setModelId} />
            <ImageDropzone onFileSelect={handleFileSelect} disabled={loading} />
          </>
        ) : (
          <div className="border border-border bg-surface overflow-hidden">
            <img
              src={displayImage!}
              alt="Prediction result"
              className="w-full block"
            />
          </div>
        )}

        {/* Controls */}
        {preview && (
          <div className="flex flex-wrap items-center gap-4 border border-border bg-surface p-4">
            <div className="flex items-center gap-3 flex-1 min-w-[200px]">
              <label className="text-xs text-text-muted uppercase tracking-wide whitespace-nowrap">
                Threshold
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-10 text-right">
                {threshold.toFixed(2)}
              </span>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handlePredict}
                disabled={loading}
                className="flex items-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                {loading ? "Analyzing..." : "Analyze"}
              </button>
              <button
                onClick={handleClear}
                disabled={loading}
                className="flex items-center gap-2 bg-surface-alt hover:bg-border text-text-secondary px-4 py-2 text-sm transition-colors disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                Clear
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="border border-danger/30 bg-danger/10 text-danger p-4 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2 space-y-4">
        {result ? (
          <>
            <div className="border border-border bg-surface p-5">
              <p className="text-xs text-text-muted uppercase tracking-wide mb-4">
                Summary
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-2xl font-bold">
                    {result.detections.length}
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">Detections</p>
                </div>
                <div>
                  <p className="text-2xl font-bold">
                    {result.inference_time_ms.toFixed(0)}
                    <span className="text-sm font-normal text-text-muted">
                      ms
                    </span>
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">Inference</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-border">
                <div>
                  <p className="text-xs text-text-muted">Model</p>
                  <p className="text-sm font-mono mt-0.5">
                    {result.model_version}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-text-muted">Image</p>
                  <p className="text-sm font-mono mt-0.5">
                    {result.image_size[0]}&times;{result.image_size[1]}
                  </p>
                </div>
              </div>
            </div>

            <ClassSummary detections={result.detections} />

            <div className="space-y-2">
              {result.detections.map((det, i) => (
                <DetectionCard key={i} detection={det} index={i} />
              ))}
            </div>

            {result.detections.length === 0 && (
              <div className="border border-border bg-surface p-6 text-center">
                <p className="text-text-muted text-sm">
                  No objects detected above the confidence threshold
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="border border-border bg-surface p-8 text-center min-h-[200px] flex flex-col items-center justify-center">
            <ImageIcon className="h-8 w-8 text-text-muted mb-3" />
            <p className="text-text-muted text-sm">
              {preview
                ? 'Click "Analyze" to run predictions'
                : "Upload an image to get started"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   BATCH IMAGE MODE
   ═══════════════════════════════════════════════════════════════════════ */

function BatchMode() {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<BatchPredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [modelId, setModelId] = useState<number | undefined>(undefined);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const handlePredict = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictBatch(files, threshold, modelId);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-6 space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <BatchDropzone
            onFilesSelect={setFiles}
            disabled={loading}
            files={files}
          />
        </div>
        <div className="space-y-4">
          <ModelSelector value={modelId} onChange={setModelId} />
          <div>
            <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
              Threshold
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-10 text-right">
                {threshold.toFixed(2)}
              </span>
            </div>
          </div>
          <button
            onClick={handlePredict}
            disabled={loading || files.length === 0}
            className="w-full flex items-center justify-center gap-2 bg-accent hover:bg-accent-hover text-white px-4 py-2.5 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Zap className="h-4 w-4" />
            )}
            {loading
              ? `Analyzing ${files.length} images...`
              : `Analyze ${files.length} image${files.length !== 1 ? "s" : ""}`}
          </button>
        </div>
      </div>

      {error && (
        <div className="border border-danger/30 bg-danger/10 text-danger p-4 text-sm">
          {error}
        </div>
      )}

      {/* Batch results */}
      {result && (
        <div className="space-y-4">
          {/* Summary bar */}
          <div className="border border-border bg-surface p-4 flex flex-wrap items-center gap-6">
            <div>
              <p className="text-xs text-text-muted">Images</p>
              <p className="text-lg font-bold">{result.results.length}</p>
            </div>
            <div>
              <p className="text-xs text-text-muted">Total Detections</p>
              <p className="text-lg font-bold">
                {result.results.reduce(
                  (sum, r) => sum + r.detections.length,
                  0,
                )}
              </p>
            </div>
            <div>
              <p className="text-xs text-text-muted">Total Time</p>
              <p className="text-lg font-bold">
                {result.total_time_ms.toFixed(0)}
                <span className="text-sm font-normal text-text-muted">ms</span>
              </p>
            </div>
            <div>
              <p className="text-xs text-text-muted">Model</p>
              <p className="text-sm font-mono">{result.model_version}</p>
            </div>
          </div>

          {/* Results grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {result.results.map((item, i) => (
              <BatchResultCard
                key={i}
                item={item}
                index={i}
                expanded={expandedIndex === i}
                onToggle={() =>
                  setExpandedIndex(expandedIndex === i ? null : i)
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function BatchResultCard({
  item,
  index,
  expanded,
  onToggle,
}: {
  item: BatchItemResult;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="border border-border bg-surface overflow-hidden">
      {item.annotated_image && (
        <img
          src={`data:image/png;base64,${item.annotated_image}`}
          alt={item.filename}
          className="w-full block"
        />
      )}
      <div className="p-3">
        <div className="flex items-center justify-between">
          <p className="text-xs font-mono text-text-muted truncate flex-1">
            {item.filename}
          </p>
          <div className="flex items-center gap-3 text-xs shrink-0 ml-2">
            <span className="font-bold">{item.detections.length} det.</span>
            <span className="text-text-muted">
              {item.inference_time_ms.toFixed(0)}ms
            </span>
          </div>
        </div>

        {item.detections.length > 0 && (
          <button
            onClick={onToggle}
            className="flex items-center gap-1 text-xs text-accent mt-2 hover:underline"
          >
            {expanded ? (
              <ChevronUp className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
            {expanded ? "Hide" : "Show"} details
          </button>
        )}

        {expanded && (
          <div className="mt-2 space-y-1">
            {item.detections.map((det, j) => {
              const color = CLASS_COLORS[det.class_name] || "#71717a";
              return (
                <div key={j} className="flex items-center gap-2 text-xs">
                  <span
                    className="w-2 h-2 shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="flex-1">{det.class_name}</span>
                  <span className="font-mono text-text-muted">
                    {(det.score * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Shared Sub-components ───────────────────────────────────────────── */

function ClassSummary({ detections }: { detections: Detection[] }) {
  const counts: Record<string, number> = {};
  detections.forEach((d) => {
    counts[d.class_name] = (counts[d.class_name] || 0) + 1;
  });

  if (Object.keys(counts).length === 0) return null;

  return (
    <div className="border border-border bg-surface p-5">
      <p className="text-xs text-text-muted uppercase tracking-wide mb-3">
        Classes Found
      </p>
      <div className="flex flex-wrap gap-2">
        {Object.entries(counts).map(([cls, count]) => (
          <span
            key={cls}
            className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium"
            style={{
              backgroundColor: `${CLASS_COLORS[cls]}20`,
              color: CLASS_COLORS[cls],
            }}
          >
            <span
              className="w-2 h-2 inline-block"
              style={{ backgroundColor: CLASS_COLORS[cls] }}
            />
            {cls} ({count})
          </span>
        ))}
      </div>
    </div>
  );
}

function DetectionCard({
  detection,
  index,
}: {
  detection: Detection;
  index: number;
}) {
  const color = CLASS_COLORS[detection.class_name] || "#71717a";
  const pct = (detection.score * 100).toFixed(1);

  return (
    <div className="border border-border bg-surface p-4">
      <div className="flex items-center gap-3">
        <span
          className="w-3 h-3 shrink-0"
          style={{ backgroundColor: color }}
        />
        <span className="font-medium text-sm capitalize flex-1">
          {detection.class_name}
        </span>
        <span className="text-xs text-text-muted">#{index + 1}</span>
      </div>

      <div className="mt-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-text-muted">Confidence</span>
          <span className="font-mono font-medium">{pct}%</span>
        </div>
        <div className="h-1.5 bg-background w-full">
          <div
            className="h-full transition-all duration-300"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
      </div>

      <div className="mt-2 text-[11px] text-text-muted font-mono">
        bbox: [{detection.bbox.map((v) => v.toFixed(0)).join(", ")}]
      </div>
    </div>
  );
}
