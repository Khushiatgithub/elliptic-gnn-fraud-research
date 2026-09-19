/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        lab: {
          bg: '#080a10',
          surface: '#0f1320',
          card: '#141a2b',
          border: '#1f2940',
          borderLight: '#2a3754',
          muted: '#627294',
          text: '#f1f5f9',
          heading: '#ffffff',
          illicit: '#f43f5e',
          licit: '#10b981',
          unknown: '#64748b',
          accent: '#6366f1',
          accentCyan: '#38bdf8',
          accentGold: '#f59e0b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Geist Mono', 'JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
