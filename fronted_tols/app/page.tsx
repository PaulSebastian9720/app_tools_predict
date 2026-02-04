"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ScanSearch, GraduationCap, Activity, Cpu, HardDrive, Layers } from "lucide-react";
import { healthCheck, getDatasets, getJobs, getModels } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [stats, setStats] = useState({ datasets: 0, jobs: 0, models: 0 });

  useEffect(() => {
    healthCheck()
      .then((h) => {
        setHealth(h);
        setConnected(true);
      })
      .catch(() => setConnected(false));

    Promise.allSettled([
      getDatasets().then((r) => r.total),
      getJobs().then((r) => r.total),
      getModels().then((r) => r.total),
    ]).then(([d, j, m]) => {
      setStats({
        datasets: d.status === "fulfilled" ? d.value : 0,
        jobs: j.status === "fulfilled" ? j.value : 0,
        models: m.status === "fulfilled" ? m.value : 0,
      });
    });
  }, []);

  return (
    <div className="p-8 max-w-5xl">
      <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
      <p className="text-text-secondary text-sm mt-1">
        Tool detection system overview
      </p>

      {/* API Status */}
      <div className="border border-border bg-surface p-5 mt-8">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="h-4 w-4 text-text-muted" />
          <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
            System Status
          </span>
        </div>

        <div className="flex items-center gap-2 mb-5">
          <div
            className={`w-2 h-2 ${
              connected === null
                ? "bg-text-muted"
                : connected
                  ? "bg-success"
                  : "bg-danger"
            }`}
          />
          <span className="text-sm">
            {connected === null
              ? "Checking..."
              : connected
                ? "API Connected"
                : "API Disconnected"}
          </span>
        </div>

        {health && (
          <div className="grid grid-cols-3 gap-6">
            <div>
              <p className="text-xs text-text-muted uppercase tracking-wide">
                Model
              </p>
              <p className="font-mono text-sm mt-1">
                {health.model_version || "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-text-muted uppercase tracking-wide">
                Device
              </p>
              <p className="font-mono text-sm mt-1">{health.device}</p>
            </div>
            <div>
              <p className="text-xs text-text-muted uppercase tracking-wide">
                Status
              </p>
              <p className="font-mono text-sm mt-1">
                {health.model_loaded ? "Ready" : "No model loaded"}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6">
        <StatCard icon={HardDrive} label="Datasets" value={stats.datasets} />
        <StatCard icon={Cpu} label="Training Jobs" value={stats.jobs} />
        <StatCard icon={Layers} label="Models" value={stats.models} />
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <Link
          href="/predict"
          className="border border-border bg-surface p-6 hover:bg-surface-alt transition-colors"
        >
          <div className="flex items-center gap-3 mb-2">
            <ScanSearch className="h-5 w-5 text-accent" />
            <h3 className="font-semibold">Predict</h3>
          </div>
          <p className="text-sm text-text-secondary">
            Upload an image for tool detection analysis
          </p>
        </Link>

        <Link
          href="/training"
          className="border border-border bg-surface p-6 hover:bg-surface-alt transition-colors"
        >
          <div className="flex items-center gap-3 mb-2">
            <GraduationCap className="h-5 w-5 text-accent" />
            <h3 className="font-semibold">Training</h3>
          </div>
          <p className="text-sm text-text-secondary">
            Manage datasets, training jobs, and model versions
          </p>
        </Link>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
}) {
  return (
    <div className="border border-border bg-surface p-5">
      <div className="flex items-center gap-2 mb-3">
        <Icon className="h-4 w-4 text-text-muted" />
        <p className="text-xs text-text-muted uppercase tracking-wide">
          {label}
        </p>
      </div>
      <p className="text-3xl font-bold">{value}</p>
    </div>
  );
}
