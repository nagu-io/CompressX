import PropTypes from "prop-types";
import { useNavigate } from "react-router-dom";

import StatusBadge from "../shared/StatusBadge";
import {
  extractRatioValue,
  formatDuration,
  formatPercent,
  formatRatio,
  formatSizeGb,
} from "../../utils/format";

function resolveJobTime(job) {
  if (typeof job.elapsed_seconds === "number") {
    return formatDuration(job.elapsed_seconds);
  }

  if (typeof job.total_time_minutes === "number") {
    return formatDuration(job.total_time_minutes * 60);
  }

  return "—";
}

export default function RecentJobs({ jobs, supported }) {
  const navigate = useNavigate();

  return (
    <div className="cx-card">
      <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
        RECENT JOBS
      </div>
      <div className="mt-2 text-lg font-semibold text-text-primary">
        Latest Compression Runs
      </div>

      {!supported || jobs.length === 0 ? (
        <div className="mt-8 rounded border border-dashed border-border-dark bg-bg-secondary px-4 py-8 text-center text-sm text-text-muted">
          Submit your first compression job to see results here.
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full text-left text-sm text-text-primary">
            <thead className="border-b border-border-dark text-xs uppercase tracking-[0.16em] text-text-muted">
              <tr>
                <th className="pb-3 pr-4">Model</th>
                <th className="pb-3 pr-4">Original</th>
                <th className="pb-3 pr-4">Final</th>
                <th className="pb-3 pr-4">Ratio</th>
                <th className="pb-3 pr-4">Accuracy</th>
                <th className="pb-3 pr-4">Status</th>
                <th className="pb-3">Time</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const ratioValue = extractRatioValue(
                  job.compression_ratio,
                  job.original_size_gb,
                  job.final_size_gb,
                );

                return (
                  <tr
                    key={job.job_id}
                    onClick={() => navigate(`/jobs/${job.job_id}`)}
                    className="cursor-pointer border-b border-border-dark/60 transition-colors hover:bg-[#1a1a1a]"
                  >
                    <td className="py-4 pr-4">{job.model_id}</td>
                    <td className="py-4 pr-4 font-mono text-text-muted">
                      {formatSizeGb(job.original_size_gb)}
                    </td>
                    <td className="py-4 pr-4 font-mono text-text-muted">
                      {formatSizeGb(job.final_size_gb)}
                    </td>
                    <td className="py-4 pr-4 font-mono text-accent">
                      {formatRatio(ratioValue)}
                    </td>
                    <td className="py-4 pr-4 font-mono">
                      {formatPercent(job.accuracy_retention_percent)}
                    </td>
                    <td className="py-4 pr-4">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="py-4 font-mono text-text-muted">
                      {resolveJobTime(job)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

RecentJobs.propTypes = {
  jobs: PropTypes.arrayOf(PropTypes.object),
  supported: PropTypes.bool.isRequired,
};
