import PropTypes from "prop-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { extractRatioValue, formatRatio } from "../../utils/format";

function TooltipContent({ active, payload }) {
  if (!active || !payload?.length) {
    return null;
  }

  const point = payload[0].payload;
  return (
    <div className="rounded border border-border-dark bg-bg-card px-3 py-2 text-xs text-text-primary">
      <div className="font-mono text-text-muted">Job {point.job}</div>
      <div className="mt-1 font-mono text-accent">{formatRatio(point.ratio)}</div>
    </div>
  );
}

TooltipContent.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.arrayOf(PropTypes.object),
};

const placeholderData = Array.from({ length: 10 }).map((_, index) => ({
  job: index + 1,
  ratio: 0,
}));

export default function CompressionChart({ jobs, supported }) {
  const chartData = jobs
    .filter((job) => job.status === "DONE")
    .slice(0, 10)
    .reverse()
    .map((job, index) => ({
      job: index + 1,
      ratio: extractRatioValue(
        job.compression_ratio,
        job.original_size_gb,
        job.final_size_gb,
      ) || 0,
    }));

  const hasData = supported && chartData.length > 0;
  const renderedData = hasData ? chartData : placeholderData;

  return (
    <div className="cx-card relative">
      <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
        COMPRESSION RATIO TREND
      </div>
      <div className="mt-2 text-lg font-semibold text-text-primary">
        Last 10 Jobs
      </div>
      <div className="mt-6 h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={renderedData}>
            <CartesianGrid stroke="#1e1e1e" strokeDasharray="3 3" />
            <XAxis
              dataKey="job"
              stroke="#555555"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#555555"
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}x`}
            />
            <Tooltip content={<TooltipContent />} />
            <Line
              type="monotone"
              dataKey="ratio"
              stroke="#00ff88"
              strokeWidth={2}
              dot={false}
              isAnimationActive={hasData}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {!hasData ? (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-black/35 font-mono text-xs uppercase tracking-[0.18em] text-text-muted">
          No data yet
        </div>
      ) : null}
    </div>
  );
}

CompressionChart.propTypes = {
  jobs: PropTypes.arrayOf(PropTypes.object),
  supported: PropTypes.bool.isRequired,
};
