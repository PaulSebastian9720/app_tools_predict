"use client";

import { useState, useEffect } from "react";
import { getModels } from "@/lib/api";
import type { ModelResponse } from "@/lib/types";
import { Layers, Loader2 } from "lucide-react";

interface ModelSelectorProps {
  value?: number;
  onChange: (modelId: number | undefined) => void;
  label?: string;
}

export function ModelSelector({ value, onChange, label = "Model" }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModels()
      .then((r) => {
        setModels(r.models);
        // Auto-select production model if no value set
        if (value === undefined) {
          const prod = r.models.find((m) => m.environment === "production");
          if (prod) onChange(prod.id);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-text-muted text-sm">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Loading models...
      </div>
    );
  }

  return (
    <div>
      <label className="block text-xs text-text-muted uppercase tracking-wide mb-1.5">
        <span className="flex items-center gap-1.5">
          <Layers className="h-3 w-3" />
          {label}
        </span>
      </label>
      <select
        value={value ?? ""}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v ? Number(v) : undefined);
        }}
        className="w-full bg-background border border-border px-3 py-2 text-sm text-text outline-none focus:border-accent"
      >
        <option value="">Auto (production)</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name} ({m.version}){" "}
            {m.environment === "production"
              ? "- Production"
              : m.environment === "staging"
                ? "- Staging"
                : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
