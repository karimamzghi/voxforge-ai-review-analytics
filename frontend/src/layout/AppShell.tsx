import { BarChart3, Boxes, FileText, Info, Lightbulb, Menu, Network, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

const navigation = [
  { to: "/", label: "Dashboard", icon: BarChart3 },
  { to: "/topics", label: "Topics", icon: Boxes },
  { to: "/recommendations", label: "Recommendations", icon: Lightbulb },
  { to: "/report", label: "Executive report", icon: FileText },
  { to: "/architecture", label: "Architecture", icon: Network },
  { to: "/about", label: "About", icon: Info },
];

export function AppShell() {
  const [open, setOpen] = useState(false);

  return (
    <div className="page-grid min-h-screen bg-ink text-slate-100">
      <button className="fixed right-4 top-4 z-50 rounded-xl border border-line bg-panel p-2 lg:hidden" onClick={() => setOpen(!open)} aria-label="Toggle navigation">
        {open ? <X /> : <Menu />}
      </button>

      <aside className={`fixed inset-y-0 left-0 z-40 w-72 border-r border-line/70 bg-ink/95 p-5 backdrop-blur transition-transform lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
        <img src="/voxforge-logo.png" alt="VoxForge AI" className="h-24 w-full object-contain object-left" />
        <p className="mt-2 text-xs uppercase tracking-[0.22em] text-slate-500">Review intelligence</p>
        <nav className="mt-8 space-y-2">
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) => `flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${isActive ? "bg-gradient-to-r from-forgeBlue/20 to-forgeOrange/10 text-white" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}
            >
              <Icon size={18} /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="absolute bottom-6 left-5 right-5 rounded-xl border border-line bg-white/[0.03] p-4">
          <p className="text-xs font-semibold text-white">Ironhack AI Engineering</p>
          <p className="mt-1 text-xs text-slate-500">NLP · ML · FastAPI · React</p>
        </div>
      </aside>

      <main className="min-h-screen lg:pl-72">
        <div className="mx-auto max-w-7xl p-5 pt-20 sm:p-8 lg:p-10 lg:pt-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
