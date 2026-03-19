import PropTypes from "prop-types";

import { normalizeStatus } from "../../utils/format";

const STATUS_STYLES = {
  DONE: "bg-accent text-black",
  RUNNING: "animate-pulse bg-accent-warn text-black",
  QUEUED: "bg-[#333333] text-[#aaaaaa]",
  FAILED: "bg-accent-danger text-white",
  CANCELLED: "bg-[#444444] text-[#888888]",
};

export default function StatusBadge({ status, large = false }) {
  const normalized = normalizeStatus(status);

  return (
    <span
      className={[
        "inline-flex items-center rounded-full font-mono uppercase tracking-[0.18em]",
        large ? "px-4 py-2 text-sm" : "px-3 py-1 text-xs",
        STATUS_STYLES[normalized] || "bg-[#333333] text-[#aaaaaa]",
      ].join(" ")}
    >
      {normalized}
    </span>
  );
}

StatusBadge.propTypes = {
  status: PropTypes.string,
  large: PropTypes.bool,
};
