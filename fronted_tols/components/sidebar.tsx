"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, ScanSearch, GraduationCap } from "lucide-react";
import type { ComponentType } from "react";

interface NavItem {
  href: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/predict", label: "Predict", icon: ScanSearch },
  { href: "/training", label: "Training", icon: GraduationCap },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 shrink-0 border-r border-border bg-surface flex flex-col h-screen">
      <div className="px-5 py-5 border-b border-border">
        <h1 className="text-base font-bold tracking-tight">Tools Detection</h1>
        <p className="text-xs text-text-muted mt-0.5">YOLOv8 Segmentation</p>
      </div>

      <nav className="flex-1 py-3">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/"
              ? pathname === "/"
              : pathname.startsWith(href);

          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                active
                  ? "text-text bg-surface-alt border-l-2 border-accent"
                  : "text-text-secondary hover:text-text hover:bg-surface-alt border-l-2 border-transparent"
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-border">
        <p className="text-[11px] text-text-muted">YOLOv8 &middot; Ultralytics</p>
      </div>
    </aside>
  );
}
