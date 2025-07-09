import React from "react";
import { useNavigate } from "react-router-dom";

export default function Header({ onRefresh }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("user");
    navigate("/");
  };

  return (
    <div className="flex justify-between items-center border-b pb-2">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <span className="text-blue-600">🛡️ EdgeGuard++ Dashboard</span>
        </h1>
        <p className="text-sm text-gray-500">Real-time anomaly detection and monitoring system</p>
      </div>

      <div className="flex items-center gap-4">
        <div className=" opacity-0 md:opacity-100 flex items-center gap-2 text-green-600 font-medium">
          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
          System Active
        </div>

        <button
          onClick={handleLogout}
          className="text-sm text-red-500 font-medium hover:underline"
        >
          Logout
        </button>

        <button
          onClick={onRefresh}
          className="text-blue-500 text-xl hover:rotate-180 transition-transform  opacity-0 md:opacity-100"
          title="Refresh"
        >
          🔄
        </button>
      </div>
    </div>
  );
}
