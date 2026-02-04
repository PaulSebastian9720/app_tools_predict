"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { CLASS_COLORS } from "@/lib/types";
import type { AnnotationBox, ImageAnnotation } from "@/lib/types";
import {
  ChevronLeft,
  ChevronRight,
  Trash2,
} from "lucide-react";

interface BboxAnnotatorProps {
  images: File[];
  classNames: { id: number; name: string }[];
  onAnnotationsChange: (annotations: ImageAnnotation[]) => void;
}

export function BboxAnnotator({
  images,
  classNames,
  onAnnotationsChange,
}: BboxAnnotatorProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [annotations, setAnnotations] = useState<Map<string, ImageAnnotation>>(
    new Map(),
  );
  const [selectedClassId, setSelectedClassId] = useState(0);
  const [drawing, setDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const redrawRef = useRef<(cw?: number, ch?: number, img?: HTMLImageElement) => void>(() => {});

  // Create object URLs for images
  useEffect(() => {
    const urls = images.map((f) => URL.createObjectURL(f));
    setImageUrls(urls);
    return () => urls.forEach(URL.revokeObjectURL);
  }, [images]);

  const currentFile = images[currentIndex];
  const currentUrl = imageUrls[currentIndex];
  const currentAnnotation = annotations.get(currentFile?.name);

  // Load image and set canvas dimensions
  useEffect(() => {
    if (!currentUrl || !canvasRef.current) return;
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      const container = containerRef.current;
      if (!container) return;

      const maxW = container.clientWidth;
      const scale = Math.min(maxW / img.width, 600 / img.height, 1);
      const displayW = Math.floor(img.width * scale);
      const displayH = Math.floor(img.height * scale);

      const canvas = canvasRef.current!;
      canvas.width = displayW;
      canvas.height = displayH;
      setImageDims({ width: img.width, height: img.height });
      redrawRef.current(displayW, displayH, img);
    };
    img.src = currentUrl;
  }, [currentUrl, currentIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  const redraw = useCallback(
    (canvasW?: number, canvasH?: number, img?: HTMLImageElement) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvasW ?? canvas.width;
      const h = canvasH ?? canvas.height;
      const image = img ?? imageRef.current;

      ctx.clearRect(0, 0, w, h);
      if (image) {
        ctx.drawImage(image, 0, 0, w, h);
      }

      // Draw existing boxes
      const ann = annotations.get(currentFile?.name);
      if (ann) {
        const scaleX = w / ann.width;
        const scaleY = h / ann.height;
        for (const box of ann.boxes) {
          const [x1, y1, x2, y2] = box.bbox;
          const color = CLASS_COLORS[box.class_name] || "#00ff00";
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(
            x1 * scaleX,
            y1 * scaleY,
            (x2 - x1) * scaleX,
            (y2 - y1) * scaleY,
          );
          ctx.fillStyle = color;
          ctx.font = "12px monospace";
          ctx.fillText(
            `${box.class_name} `,
            x1 * scaleX + 2,
            y1 * scaleY - 4,
          );
        }
      }

      // Draw current drawing box
      if (drawing) {
        const selectedClass = classNames.find((c) => c.id === selectedClassId);
        ctx.strokeStyle = CLASS_COLORS[selectedClass?.name ?? ""] || "#00ff00";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(
          startPos.x,
          startPos.y,
          currentPos.x - startPos.x,
          currentPos.y - startPos.y,
        );
        ctx.setLineDash([]);
      }
    },
    [annotations, currentFile, drawing, startPos, currentPos, selectedClassId, classNames],
  );

  // Keep redraw ref in sync so img.onload always uses latest
  redrawRef.current = redraw;

  // Redraw on annotation/drawing changes
  useEffect(() => {
    redraw();
  }, [redraw]);

  const getCanvasCoords = (e: React.MouseEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const pos = getCanvasCoords(e);
    setDrawing(true);
    setStartPos(pos);
    setCurrentPos(pos);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing) return;
    setCurrentPos(getCanvasCoords(e));
  };

  const handleMouseUp = () => {
    if (!drawing || !canvasRef.current) return;
    setDrawing(false);

    const canvas = canvasRef.current;
    const scaleX = imageDims.width / canvas.width;
    const scaleY = imageDims.height / canvas.height;

    const x1 = Math.min(startPos.x, currentPos.x) * scaleX;
    const y1 = Math.min(startPos.y, currentPos.y) * scaleY;
    const x2 = Math.max(startPos.x, currentPos.x) * scaleX;
    const y2 = Math.max(startPos.y, currentPos.y) * scaleY;

    // Ignore tiny boxes
    if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) return;

    const selectedCls = classNames.find((c) => c.id === selectedClassId);
    const className = selectedCls?.name || `class_${selectedClassId}`;
    const newBox: AnnotationBox = {
      class_id: selectedClassId,
      class_name: className,
      bbox: [
        Math.round(x1),
        Math.round(y1),
        Math.round(x2),
        Math.round(y2),
      ],
    };

    const updated = new Map(annotations);
    const existing = updated.get(currentFile.name) || {
      filename: currentFile.name,
      width: imageDims.width,
      height: imageDims.height,
      boxes: [],
    };
    existing.boxes = [...existing.boxes, newBox];
    updated.set(currentFile.name, existing);
    setAnnotations(updated);
    onAnnotationsChange(Array.from(updated.values()));
  };

  const deleteBox = (boxIndex: number) => {
    const updated = new Map(annotations);
    const ann = updated.get(currentFile.name);
    if (!ann) return;
    ann.boxes = ann.boxes.filter((_, i) => i !== boxIndex);
    if (ann.boxes.length === 0) {
      updated.delete(currentFile.name);
    } else {
      updated.set(currentFile.name, { ...ann });
    }
    setAnnotations(updated);
    onAnnotationsChange(Array.from(updated.values()));
  };

  if (images.length === 0) {
    return (
      <div className="border border-border bg-surface p-8 text-center">
        <p className="text-text-muted text-sm">Upload images first to annotate</p>
      </div>
    );
  }

  const totalBoxes = Array.from(annotations.values()).reduce(
    (sum, a) => sum + a.boxes.length,
    0,
  );

  return (
    <div className="space-y-4">
      {/* Navigation + class selector */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
            disabled={currentIndex === 0}
            className="p-1.5 border border-border hover:bg-surface-alt disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="text-sm font-mono min-w-[80px] text-center">
            {currentIndex + 1} / {images.length}
          </span>
          <button
            onClick={() =>
              setCurrentIndex((i) => Math.min(images.length - 1, i + 1))
            }
            disabled={currentIndex === images.length - 1}
            className="p-1.5 border border-border hover:bg-surface-alt disabled:opacity-30 transition-colors"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        <select
          value={selectedClassId}
          onChange={(e) => setSelectedClassId(Number(e.target.value))}
          className="bg-background border border-border px-3 py-1.5 text-sm text-text outline-none focus:border-accent"
        >
          {classNames.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>

        <span className="text-xs text-text-muted ml-auto">
          {totalBoxes} annotation{totalBoxes !== 1 && "s"} across{" "}
          {annotations.size} image{annotations.size !== 1 && "s"}
        </span>
      </div>

      {/* Canvas area */}
      <div ref={containerRef} className="border border-border bg-black/5 overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            if (drawing) setDrawing(false);
          }}
          className="cursor-crosshair block mx-auto"
        />
      </div>

      {/* Filename */}
      <p className="text-xs text-text-muted font-mono truncate">
        {currentFile?.name}
      </p>

      {/* Annotation list for current image */}
      {currentAnnotation && currentAnnotation.boxes.length > 0 && (
        <div className="border border-border bg-surface">
          <div className="px-4 py-2 border-b border-border">
            <p className="text-xs text-text-muted uppercase tracking-wide">
              Annotations ({currentAnnotation.boxes.length})
            </p>
          </div>
          <div className="divide-y divide-border">
            {currentAnnotation.boxes.map((box, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-4 py-2 text-sm"
              >
                <span
                  className="w-3 h-3 shrink-0"
                  style={{
                    backgroundColor: CLASS_COLORS[box.class_name] || "#888",
                  }}
                />
                <span className="flex-1">{box.class_name}</span>
                <span className="text-xs text-text-muted font-mono">
                  [{box.bbox.map((v) => Math.round(v)).join(", ")}]
                </span>
                <button
                  onClick={() => deleteBox(i)}
                  className="text-text-muted hover:text-danger transition-colors"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
