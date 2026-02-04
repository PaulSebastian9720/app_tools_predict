"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FolderOpen } from "lucide-react";

interface BatchDropzoneProps {
  onFilesSelect: (files: File[]) => void;
  disabled?: boolean;
  files?: File[];
}

export function BatchDropzone({ onFilesSelect, disabled, files = [] }: BatchDropzoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFilesSelect(accepted);
    },
    [onFilesSelect],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    disabled,
    multiple: true,
  });

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);
  const sizeStr =
    totalSize > 1024 * 1024
      ? `${(totalSize / 1024 / 1024).toFixed(1)} MB`
      : `${(totalSize / 1024).toFixed(0)} KB`;

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed flex flex-col items-center justify-center p-12 text-center cursor-pointer transition-colors min-h-[240px] ${
          isDragActive
            ? "border-accent bg-accent/5"
            : "border-border hover:border-border-hover"
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <input {...getInputProps()} />
        <FolderOpen className="h-10 w-10 text-text-muted mb-4" />
        <p className="text-text-secondary text-sm">
          {isDragActive
            ? "Drop the images here"
            : "Drag & drop images, or click to select"}
        </p>
        <p className="text-text-muted text-xs mt-2">
          JPG, PNG, WebP — multiple files supported
        </p>
      </div>

      {files.length > 0 && (
        <div className="mt-3 border border-border bg-surface p-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm">
            <Upload className="h-4 w-4 text-text-muted" />
            <span className="font-medium">
              {files.length} image{files.length !== 1 && "s"}
            </span>
            <span className="text-text-muted">({sizeStr})</span>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFilesSelect([]);
            }}
            className="text-xs text-text-muted hover:text-danger transition-colors"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
