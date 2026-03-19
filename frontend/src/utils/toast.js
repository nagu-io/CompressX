export function pushToast({ message, tone = "error" }) {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent("compressx:toast", {
      detail: {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        message,
        tone,
      },
    }),
  );
}
