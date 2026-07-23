/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#070b17",
        panel: "#101827",
        line: "#23314d",
        forgeBlue: "#11b8f2",
        forgeOrange: "#ff8b37",
        forgePink: "#ef4770"
      },
      boxShadow: {
        glow: "0 0 40px rgba(17,184,242,0.16)"
      }
    }
  },
  plugins: []
};
