import PropTypes from "prop-types";

export default function ConfigEditor({ yaml }) {
  return (
    <div className="cx-card h-full">
      <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
        CONFIG PREVIEW
      </div>
      <div className="mt-4 rounded border border-border-dark bg-black">
        <pre className="min-h-[720px] overflow-x-auto px-4 py-4 font-mono text-xs leading-6 text-accent">
          {yaml}
        </pre>
      </div>
    </div>
  );
}

ConfigEditor.propTypes = {
  yaml: PropTypes.string.isRequired,
};
