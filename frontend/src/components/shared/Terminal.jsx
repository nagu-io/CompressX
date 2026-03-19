import PropTypes from "prop-types";
import { useEffect, useMemo, useRef } from "react";

import { formatClock, hasTimestampPrefix, safeArray } from "../../utils/format";

export default function Terminal({ lines, isRunning }) {
  const viewportRef = useRef(null);

  useEffect(() => {
    if (viewportRef.current) {
      viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
    }
  }, [lines]);

  const fallbackTime = useMemo(() => formatClock(), [lines]);

  return (
    <div className="rounded border border-border-dark bg-black">
      <div className="border-b border-border-dark px-4 py-3 font-mono text-xs uppercase tracking-[0.22em] text-text-muted">
        Terminal
      </div>
      <div
        ref={viewportRef}
        className="h-[280px] overflow-y-auto px-4 py-4 font-mono text-xs leading-6 text-accent"
      >
        {safeArray(lines).length === 0 ? (
          <div className="text-text-muted">[--:--:--] Waiting for backend logs...</div>
        ) : (
          safeArray(lines).map((line, index) => {
            const renderedLine = hasTimestampPrefix(line)
              ? line
              : `[${fallbackTime}] ${line}`;

            return (
              <div key={`${renderedLine}-${index}`} className="whitespace-pre-wrap break-words">
                {renderedLine}
              </div>
            );
          })
        )}
        {isRunning ? <span className="terminal-cursor ml-1 inline-block">█</span> : null}
      </div>
    </div>
  );
}

Terminal.propTypes = {
  lines: PropTypes.arrayOf(PropTypes.string),
  isRunning: PropTypes.bool,
};
