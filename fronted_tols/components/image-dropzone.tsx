"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";

interface ImageDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function ImageDropzone({ onFileSelect, disabled }: ImageDropzoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFileSelect(accepted[0]);
    },
    [onFileSelect],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed flex flex-col items-center justify-center p-16 text-center cursor-pointer transition-colors min-h-[360px] ${
        isDragActive
          ? "border-accent bg-accent/5"
          : "border-border hover:border-border-hover"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input {...getInputProps()} />
      <Upload className="h-10 w-10 text-text-muted mb-4" />
      <p className="text-text-secondary text-sm">
        {isDragActive
          ? "Drop the image here"
          : "Drag & drop an image, or click to select"}
      </p>
      <p className="text-text-muted text-xs mt-2">JPG, PNG, WebP</p>
    </div>
  );
}
