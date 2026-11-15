/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        /* ── Spider-Verse Palette ── */
        noir: {
          950: "#08080c",
          900: "#0d0d14",
          800: "#12121f",
          700: "#1a1a2e",
          600: "#252540",
        },
        neon: {
          magenta: "#ff2d7b",
          cyan: "#00f0ff",
          yellow: "#ffe156",
          purple: "#b24dff",
          green: "#39ff14",
        },
        holo: {
          pink: "#ff6ec7",
          blue: "#4d9fff",
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Fira Code"', "monospace"],
        body: ['"Inter"', "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
        "scan-line": "scanLine 4s linear infinite",
        "glitch-x": "glitchX 0.3s ease-in-out",
        float: "float 6s ease-in-out infinite",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        scanLine: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        glitchX: {
          "0%, 100%": { transform: "translateX(0)" },
          "25%": { transform: "translateX(-3px)" },
          "75%": { transform: "translateX(3px)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(255,45,123,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,45,123,0.03) 1px, transparent 1px)",
        "halftone":
          "radial-gradient(circle, rgba(255,45,123,0.08) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "40px 40px",
        dots: "20px 20px",
      },
    },
  },
  plugins: [],
};
