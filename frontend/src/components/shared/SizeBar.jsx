import PropTypes from "prop-types";
import { motion } from "framer-motion";

import { formatSizeGb, numberOrNull } from "../../utils/format";

function BarRow({ label, value, maxValue, tone, animated }) {
  const normalized = numberOrNull(value) || 0;
  const width = maxValue > 0 ? `${(normalized / maxValue) * 100}%` : "0%";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between font-mono text-xs text-text-muted">
        <span>{label}</span>
        <span className="text-text-primary">{formatSizeGb(value)}</span>
      </div>
      <div className="h-3 overflow-hidden rounded bg-[#111111]">
        {animated ? (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className={`h-full rounded ${tone}`}
          />
        ) : (
          <div className={`h-full rounded ${tone}`} style={{ width }} />
        )}
      </div>
    </div>
  );
}

BarRow.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number,
  maxValue: PropTypes.number.isRequired,
  tone: PropTypes.string.isRequired,
  animated: PropTypes.bool.isRequired,
};

export default function SizeBar({ originalGb, finalGb }) {
  const maxValue = Math.max(numberOrNull(originalGb) || 0, numberOrNull(finalGb) || 0, 1);

  return (
    <div className="space-y-4">
      <BarRow
        label="Original"
        value={originalGb}
        maxValue={maxValue}
        tone="bg-[#333333]"
        animated={false}
      />
      <BarRow
        label="Final"
        value={finalGb}
        maxValue={maxValue}
        tone="bg-accent"
        animated
      />
    </div>
  );
}

SizeBar.propTypes = {
  originalGb: PropTypes.number,
  finalGb: PropTypes.number,
};
