import PropTypes from "prop-types";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const ToastContext = createContext(null);
const TOAST_EVENT = "compressx:toast";

function buildToast(payload) {
  return {
    id: payload.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    tone: payload.tone || "error",
    message: payload.message || "Unknown notification",
  };
}

export function pushToast(payload) {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent(TOAST_EVENT, {
      detail: buildToast(payload),
    }),
  );
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    function handleToast(event) {
      const toast = buildToast(event.detail || {});
      setToasts((current) => [...current, toast]);

      window.setTimeout(() => {
        setToasts((current) => current.filter((item) => item.id !== toast.id));
      }, 4000);
    }

    window.addEventListener(TOAST_EVENT, handleToast);
    return () => {
      window.removeEventListener(TOAST_EVENT, handleToast);
    };
  }, []);

  const value = useMemo(
    () => ({
      showError(message) {
        pushToast({ message, tone: "error" });
      },
      showSuccess(message) {
        pushToast({ message, tone: "success" });
      },
    }),
    [],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-3 px-4 sm:px-0">
        <AnimatePresence>
          {toasts.map((toast) => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
              className={[
                "pointer-events-auto rounded border border-border-dark bg-bg-card px-4 py-3 text-sm text-text-primary shadow-panel",
                toast.tone === "success"
                  ? "border-l-[3px] border-l-accent"
                  : "border-l-[3px] border-l-accent-danger",
              ].join(" ")}
            >
              {toast.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

ToastProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within ToastProvider.");
  }
  return context;
}
