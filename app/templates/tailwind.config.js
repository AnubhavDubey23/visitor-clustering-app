module.exports = {
  darkMode: "class",
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
    "./node_modules/flowbite/**/*.js"
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
          800: "#1E40AF",
          900: "#1E3A8A",
        },
        // Cluster colors matching Plotly
        cluster: {
          0: "#FF6384",
          1: "#36A2EB",
          2: "#FFCE56",
          3: "#4BC0C0",
          4: "#9966FF",
          5: "#FF9F40",
          6: "#8AC926",
          7: "#1982C4",
          8: "#6A4C93",
          9: "#F15BB5"
        }
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      boxShadow: {
        soft: "0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)",
        plot: "0 1px 3px rgba(0, 0, 0, 0.1)",
        'plot-dark': "0 1px 3px rgba(0, 0, 0, 0.3)",
      },
      animation: {
        "bounce-slow": "bounce 3s infinite",
        "fade-in": "fadeIn 0.3s ease-out",
        spin: "spin 1s linear infinite",
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        spin: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" }
        }
      },
      minHeight: {
        plot: "400px",
        'plot-mobile': "300px",
      },
    },
  },
  plugins: [
    require('flowbite/plugin')
  ],
}