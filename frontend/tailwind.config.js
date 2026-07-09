/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ocean: {
          navy:    '#03071e',
          dark:    '#050b14',
          teal:    '#0d4f6b',
          seafoam: '#2dd4bf',
          coral:   '#fb7185',
          gold:    '#fbbf24',
          purple:  '#c084fc',
          orange:  '#f97316',
          green:   '#34d399',
          blue:    '#60a5fa',
          indigo:  '#818cf8',
          pink:    '#f472b6',
        }
      },
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        body:    ['DM Sans', 'sans-serif'],
        data:    ['Fira Code', 'monospace'],
      },
      animation: {
        'blob':        'blob 7s infinite',
        'pulse-glow':  'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float':       'float 6s ease-in-out infinite',
        'shimmer':     'shimmer 2.5s linear infinite',
        'wave-drift':  'wave-drift 8s ease-in-out infinite',
      },
      keyframes: {
        blob: {
          '0%':   { transform: 'translate(0px, 0px) scale(1)' },
          '33%':  { transform: 'translate(30px, -50px) scale(1.1)' },
          '66%':  { transform: 'translate(-20px, 20px) scale(0.9)' },
          '100%': { transform: 'translate(0px, 0px) scale(1)' },
        },
        'pulse-glow': {
          '0%, 100%': { opacity: '1', filter: 'drop-shadow(0 0 10px rgba(45, 212, 191, 0.5))' },
          '50%':      { opacity: '.7', filter: 'drop-shadow(0 0 20px rgba(45, 212, 191, 0.8))' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-12px)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        'wave-drift': {
          '0%, 100%': { transform: 'translateX(0) translateY(0)' },
          '50%':      { transform: 'translateX(10px) translateY(-5px)' },
        },
      }
    },
  },
  plugins: [],
}
