import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import { useNavigate } from "react-router-dom";
import Header from "../components/dashboard/Header";

export default function RTSPForm() {
  const [rtspUrl, setRtspUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      const user = JSON.parse(localStorage.getItem("user"));
      const res = await axios.post("http://localhost:5000/api/stream/start", {
        email: user?.email,
        rtsp: rtspUrl,
      });
      toast.success(res.data.message);
      setRtspUrl("");
    } catch (err) {
      toast.error("Error: " + (err.response?.data?.message || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-6 py-4 bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-white transition-colors duration-300">
        <div className="">
          <Header/>
        </div>
        <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12 bg-[#EADFD1] dark:bg-[#212A3B] transition-colors duration-300">
  <div className="bg-[#FFF7EB] dark:bg-[#3C3E7D] shadow-2xl rounded-xl p-8 w-full max-w-xl border border-white/30 backdrop-blur-md transition-colors duration-300 animate-float">
    
    <h2 className="text-2xl font-black mb-4 text-center text-[#2564EA] dark:text-[#60A5F9]">
      📡 Start RTSP Stream
    </h2>

    <form onSubmit={handleSubmit} className="space-y-4 w-full">
      <input
        type="text"
        placeholder="Enter RTSP stream URL"
        value={rtspUrl}
        onChange={(e) => setRtspUrl(e.target.value)}
        required
        className="w-full border border-gray-300 px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-800 dark:text-white dark:bg-[#4D4F9F] dark:border-transparent transition-colors"
      />

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition disabled:opacity-50"
      >
        {loading ? "Starting..." : "Start Stream"}
      </button>

      <button
        type="button"
        onClick={() => navigate("/dashboard")}
        className="w-full mt-2 border-2 border-gray-500 dark:border-gray-200 text-gray-800 dark:text-white py-2 rounded-md hover:bg-gray-100 dark:hover:bg-[#4D4F9F] transition"
      >
        ⬅️ Back to Dashboard
      </button>
    </form>
  </div>
</div>

    </div>
    
  );
}
