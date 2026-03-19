import PropTypes from "prop-types";
import { motion } from "framer-motion";

import { formatDuration } from "../../utils/format";

function StageIcon({ status }) {
  if (status === "COMPLETE") {
    return <span className="font-mono text-lg text-accent">●</span>;
  }

  if (status === "RUNNING") {
    return <span className="spinner-arc h-5 w-5" />;
  }

  if (status === "FAILED") {
    return <span className="font-mono text-lg text-accent-danger">✕</span>;
  }

  return <span className="font-mono text-lg text-[#333333]">○</span>;
}

StageIcon.propTypes = {
  status: PropTypes.string.isRequired,
};

export default function StageProgress({ stages }) {
  return (
    <div className="cx-card">
      <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
        STAGE PROGRESS
      </div>
      <div className="mt-6 space-y-5">
        {stages.map((stage, index) => (
          <motion.div
            key={stage.key}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.08, duration: 0.2 }}
            className="relative flex gap-4"
          >
            <div className="flex flex-col items-center">
              <div className="flex h-6 w-6 items-center justify-center">
                <StageIcon status={stage.status} />
              </div>
              {index < stages.length - 1 ? (
                <div className="mt-2 h-12 w-px bg-border-dark" />
              ) : null}
            </div>
            <div className="flex-1 pb-4">
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="font-mono text-sm text-text-primary">
                    Stage {index + 1}: {stage.label}
                  </div>
                  <div className="mt-1 font-mono text-xs uppercase tracking-[0.18em] text-text-muted">
                    {stage.status}
                  </div>
                </div>
                <div className="font-mono text-xs text-text-muted">
                  {formatDuration(stage.durationSeconds)}
                </div>
              </div>
              {stage.status === "RUNNING" ? (
                <div className="stage-runner mt-3 h-[2px] rounded bg-accent/20" />
              ) : null}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

StageProgress.propTypes = {
  stages: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      status: PropTypes.string.isRequired,
      durationSeconds: PropTypes.number,
    }),
  ).isRequired,
};
