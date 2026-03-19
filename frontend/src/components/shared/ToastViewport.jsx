import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

const toneClasses = {
  success: "border-accent/40 bg-accent/10 text-accent",
  error: "border-danger/40 bg-danger/10 text-white",
};

export default function ToastViewport() {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    function handleToast(event) {
      const nextToast = event.detail;
      setToasts((current) => [...current, nextToast]);

      window.setTimeout(() => {
        setToasts((current) =>
          current.filter((toast) => toast.id !== nextToast.id),
        );
      }, 4000);
    }

    window.addEventListener("compressx:toast", handleToast);
    return () => {
      window.removeEventListener("compressx:toast", handleToast);
    };
  }, []);

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-3">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            animate={{ opacity: 1, y: 0 }}
            className={[
              "pointer-events-auto rounded-2xl border px-4 py-3 shadow-panel backdrop-blur-xl",
              toneClasses[toast.tone] || toneClasses.error,
            ].join(" ")}
            exit={{ opacity: 0, y: 12 }}
            initial={{ opacity: 0, y: 12 }}
          >
            <p className="font-medium">{toast.message}</p>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
