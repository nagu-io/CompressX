/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        "bg-primary": "#0a0a0a",
        "bg-secondary": "#111111",
        "bg-card": "#161616",
        accent: "#00ff88",
        "accent-warn": "#ffaa00",
        "accent-danger": "#ff4444",
        "text-primary": "#f0f0f0",
        "text-muted": "#555555",
        "border-dark": "#1e1e1e",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      boxShadow: {
        panel: "0 12px 48px rgba(0, 0, 0, 0.35)",
      },
      keyframes: {
        blink: {
          "0%, 49%": { opacity: "1" },
          "50%, 100%": { opacity: "0" },
        },
      },
      animation: {
        blink: "blink 1s step-end infinite",
      },
    },
  },
  plugins: [],
};
