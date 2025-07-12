import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BsSunFill, BsMoonFill } from "react-icons/bs";

export default function Header({ onRefresh }) {
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("theme") === "dark");

  // Apply theme on mount and toggle
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const toggleTheme = () => setDarkMode(prev => !prev);

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("user");
    navigate("/");
  };

  return (
    <div className="flex justify-between items-center border-b pb-2">
      <div onClick={() => navigate("/dashboard")} className="cursor-pointer">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <span className="text-blue-600">🛡️ EdgeGuard Plus</span>
        </h1>
      </div>

      <div className="flex items-center gap-4">
        <div className="opacity-0 md:opacity-100 flex items-center gap-2 text-green-600 font-medium">
          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
          System Active
        </div>

        <button
          onClick={toggleTheme}
          title="Toggle Theme"
          className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 hover:scale-110 transition"
        >
          {darkMode ? <BsSunFill className="text-yellow-400" /> : <BsMoonFill />}
        </button>

        <button
          onClick={handleLogout}
          className="text-sm text-red-500 font-medium hover:underline"
        >
          Logout
        </button>
      </div>
    </div>
  );
}
