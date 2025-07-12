/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      backgroundImage: {
        'soft-gradient': 'linear-gradient(to bottom, #ffe5db, #e6eafc)',
        'auth-gradient': 'linear-gradient(135deg, #1c0140, #28044e, #320a5e, #3f0f6b, #4b146d)',
        'glass-dark': 'linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))',
      },
      keyframes: {
        glitch: {
          '0%': { clipPath: 'inset(20% 0 50% 0)' },
          '5%': { clipPath: 'inset(10% 0 60% 0)' },
          '10%': { clipPath: 'inset(15% 0 55% 0)' },
          '15%': { clipPath: 'inset(25% 0 35% 0)' },
          '20%': { clipPath: 'inset(30% 0 40% 0)' },
          '25%': { clipPath: 'inset(40% 0 20% 0)' },
          '30%': { clipPath: 'inset(10% 0 60% 0)' },
          '35%': { clipPath: 'inset(15% 0 55% 0)' },
          '40%': { clipPath: 'inset(25% 0 35% 0)' },
          '45%': { clipPath: 'inset(30% 0 40% 0)' },
          '50%': { clipPath: 'inset(20% 0 50% 0)' },
          '55%': { clipPath: 'inset(10% 0 60% 0)' },
          '60%': { clipPath: 'inset(15% 0 55% 0)' },
          '65%': { clipPath: 'inset(25% 0 35% 0)' },
          '70%': { clipPath: 'inset(30% 0 40% 0)' },
          '75%': { clipPath: 'inset(40% 0 20% 0)' },
          '80%': { clipPath: 'inset(20% 0 50% 0)' },
          '85%': { clipPath: 'inset(10% 0 60% 0)' },
          '90%': { clipPath: 'inset(15% 0 55% 0)' },
          '95%': { clipPath: 'inset(25% 0 35% 0)' },
          '100%': { clipPath: 'inset(30% 0 40% 0)' },
        },
        float: {
      '0%': { transform: 'translate(0, 0)' },
      '25%': { transform: 'translate(10px, -10px)' },
      '50%': { transform: 'translate(0, -20px)' },
      '75%': { transform: 'translate(-10px, -10px)' },
      '100%': { transform: 'translate(0, 0)' },
    },
      },
      animation: {
        'glitch-after': 'glitch var(--after-duration) infinite linear alternate-reverse',
        'glitch-before': 'glitch var(--before-duration) infinite linear alternate-reverse',
        float: 'float 4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
