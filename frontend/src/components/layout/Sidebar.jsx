import PropTypes from "prop-types";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "grid" },
  { to: "/compress", label: "New Compression", icon: "zap" },
  { to: "/settings", label: "Settings", icon: "settings" },
];

function GridIcon() {
  return (
    <span className="grid h-4 w-4 grid-cols-2 gap-0.5">
      {Array.from({ length: 4 }).map((_, index) => (
        <span key={index} className="rounded-sm bg-current" />
      ))}
    </span>
  );
}

GridIcon.propTypes = {};

function ZapIcon() {
  return (
    <span className="relative block h-4 w-4">
      <span className="absolute left-1 top-0 h-2 w-1 -skew-x-[20deg] bg-current" />
      <span className="absolute left-0 top-2 h-2 w-1 skew-x-[20deg] bg-current" />
      <span className="absolute left-2 top-2 h-2 w-1 -skew-x-[20deg] bg-current" />
    </span>
  );
}

ZapIcon.propTypes = {};

function SettingsIcon() {
  return (
    <span className="relative block h-4 w-4 rounded-full border border-current">
      <span className="absolute inset-1 rounded-full bg-current" />
    </span>
  );
}

SettingsIcon.propTypes = {};

function NavIcon({ icon }) {
  if (icon === "grid") {
    return <GridIcon />;
  }

  if (icon === "zap") {
    return <ZapIcon />;
  }

  return <SettingsIcon />;
}

NavIcon.propTypes = {
  icon: PropTypes.string.isRequired,
};

function linkClasses({ isActive }) {
  return [
    "flex items-center gap-3 border-l-2 px-4 py-3 text-sm transition-colors duration-200",
    isActive
      ? "border-l-accent bg-[#1a1a1a] text-text-primary"
      : "border-l-transparent text-text-muted hover:bg-[#161616] hover:text-text-primary",
  ].join(" ");
}

export default function Sidebar({ open, onClose, isConnected }) {
  return (
    <>
      <div
        className={[
          "fixed inset-0 z-30 bg-black/70 transition-opacity duration-200 md:hidden",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        ].join(" ")}
        onClick={onClose}
      />
      <aside
        className={[
          "fixed inset-y-0 left-0 z-40 flex w-[240px] flex-col border-r border-border-dark bg-bg-secondary transition-transform duration-200 md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        <div className="border-b border-border-dark px-5 py-6">
          <div className="font-mono text-lg font-bold tracking-[0.18em]">
            <span className="text-accent">COMPRESS</span>
            <span className="text-white">X</span>
          </div>
          <p className="mt-2 text-xs text-text-muted">LLM Compression Engine</p>
        </div>

        <nav className="flex-1 py-4">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onClose}
              className={linkClasses}
            >
              <NavIcon icon={item.icon} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-border-dark px-5 py-4">
          <p className="flex items-center gap-2 text-xs text-text-muted">
            <span
              className={[
                "h-2 w-2 rounded-full",
                isConnected ? "bg-accent" : "bg-accent-danger",
              ].join(" ")}
            />
            {isConnected ? "API Connected" : "API Offline"}
          </p>
        </div>
      </aside>
    </>
  );
}

Sidebar.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  isConnected: PropTypes.bool.isRequired,
};
