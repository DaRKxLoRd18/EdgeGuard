import React from 'react'

const SubmitButton = ({text}) => {
  return (
    <div>
      <button
        type="submit"
        className="w-full py-2 px-4 rounded-lg text-white font-semibold flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-fuchsia-500 hover:opacity-90 transition"
        >
        {text}
        <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2"
        >
            <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 11c0-2.5 2-4.5 4.5-4.5S21 8.5 21 11v2c0 2.5-2 4.5-4.5 4.5S12 15.5 12 13v-2z"
            />
            <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 12h.01"
            />
        </svg>
        </button>

    </div>
  )
}

export default SubmitButton
