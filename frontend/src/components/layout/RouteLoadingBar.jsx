import PropTypes from "prop-types";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export default function RouteLoadingBar({ pathname }) {
  const mountedRef = useRef(false);
  const [visible, setVisible] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      return undefined;
    }

    setVisible(true);
    setProgress(14);

    const timers = [
      window.setTimeout(() => setProgress(42), 50),
      window.setTimeout(() => setProgress(76), 150),
      window.setTimeout(() => setProgress(92), 280),
      window.setTimeout(() => setProgress(100), 420),
      window.setTimeout(() => {
        setVisible(false);
        setProgress(0);
      }, 620),
    ];

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
    };
  }, [pathname]);

  return (
    <AnimatePresence>
      {visible ? (
        <motion.div
          animate={{ opacity: 1 }}
          className="pointer-events-none fixed inset-x-0 top-0 z-[80] h-1 bg-accent/5"
          exit={{ opacity: 0 }}
          initial={{ opacity: 0 }}
          key="route-loading-bar"
        >
          <motion.div
            animate={{ width: `${progress}%` }}
            className="h-full bg-accent shadow-[0_0_18px_rgba(0,255,136,0.82)]"
            transition={{ duration: 0.18, ease: "easeOut" }}
          />
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

RouteLoadingBar.propTypes = {
  pathname: PropTypes.string.isRequired,
};
