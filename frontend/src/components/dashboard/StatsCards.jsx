import PropTypes from "prop-types";
import { motion } from "framer-motion";

import {
  accuracyColorClass,
  formatPercent,
  formatRatio,
  formatSizeGb,
  numberOrNull,
} from "../../utils/format";

function GridIcon() {
  return (
    <div className="grid h-8 w-8 grid-cols-2 gap-1 text-accent">
      {Array.from({ length: 4 }).map((_, index) => (
        <span key={index} className="rounded-sm border border-current" />
      ))}
    </div>
  );
}

GridIcon.propTypes = {};

function MetricValue({ value, className }) {
  return (
    <div className={["font-mono text-3xl font-bold", className].join(" ")}>
      {value}
    </div>
  );
}

MetricValue.propTypes = {
  value: PropTypes.string.isRequired,
  className: PropTypes.string,
};

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.08,
    },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0 },
};

export default function StatsCards({ stats }) {
  const avgAccuracy = numberOrNull(stats?.average_accuracy_retention);
  const storageSaved = numberOrNull(stats?.total_storage_saved_gb);
  const cards = [
    {
      key: "jobs",
      label: "TOTAL JOBS",
      value:
        stats && Number.isFinite(Number(stats.total_jobs))
          ? String(stats.total_jobs)
          : "—",
      className: "text-text-primary",
      icon: <GridIcon />,
    },
    {
      key: "ratio",
      label: "AVG RATIO",
      value: stats ? formatRatio(stats.average_compression_ratio) : "—",
      className: "text-accent",
    },
    {
      key: "saved",
      label: "STORAGE SAVED",
      value: storageSaved === null ? "—" : formatSizeGb(storageSaved),
      className: "text-accent",
    },
    {
      key: "accuracy",
      label: "AVG ACCURACY",
      value: avgAccuracy === null ? "—" : formatPercent(avgAccuracy),
      className: accuracyColorClass(avgAccuracy),
    },
  ];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"
    >
      {cards.map((card) => (
        <motion.div key={card.key} variants={cardVariants} className="cx-card">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-xs uppercase tracking-[0.26em] text-text-muted">
                {card.label}
              </div>
              <div className="mt-4">
                <MetricValue value={card.value} className={card.className} />
              </div>
            </div>
            {card.icon || <span className="h-8 w-8 border border-border-dark" />}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}

StatsCards.propTypes = {
  stats: PropTypes.shape({
    total_jobs: PropTypes.number,
    average_compression_ratio: PropTypes.number,
    total_storage_saved_gb: PropTypes.number,
    average_accuracy_retention: PropTypes.number,
  }),
};
