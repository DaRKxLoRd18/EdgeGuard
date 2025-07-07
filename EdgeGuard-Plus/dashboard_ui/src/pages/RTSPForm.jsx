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
    <div>
        <div>
          <Header/>
        </div>
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center px-4 py-12">
        <div className="bg-white shadow-lg rounded-xl p-8 w-full max-w-xl border border-blue-200">
            <h2 className="text-2xl font-semibold mb-4 text-center text-blue-600">
            📡 Start RTSP Stream
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
            <input
                type="text"
                placeholder="Enter RTSP stream URL"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                required
                className="w-full border border-gray-300 px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
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
                className="w-full mt-2 border border-gray-400 text-gray-700 py-2 rounded-md hover:bg-gray-100 transition"
            >
                ⬅️ Back to Dashboard
            </button>
            </form>
        </div>
    </div>
    </div>
    
  );
}
