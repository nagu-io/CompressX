import PropTypes from "prop-types";

export default function TopBar({ title, onMenuClick }) {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center border-b border-border-dark bg-bg-secondary px-4 md:px-6">
      <button
        type="button"
        onClick={onMenuClick}
        className="flex h-10 w-10 items-center justify-center rounded border border-border-dark bg-bg-card text-text-primary md:hidden"
      >
        <span className="sr-only">Open sidebar</span>
        <span className="space-y-1">
          <span className="block h-0.5 w-5 bg-current" />
          <span className="block h-0.5 w-5 bg-current" />
          <span className="block h-0.5 w-5 bg-current" />
        </span>
      </button>
      <div className="ml-auto text-right">
        <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-text-muted">
          CompressX Control Plane
        </div>
        <div className="mt-1 font-mono text-sm uppercase tracking-[0.18em] text-text-primary">
          {title}
        </div>
      </div>
    </header>
  );
}

TopBar.propTypes = {
  title: PropTypes.string.isRequired,
  onMenuClick: PropTypes.func.isRequired,
};
