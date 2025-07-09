import React, { useEffect, useState } from "react";
import Header from "../components/dashboard/Header";
import StatCard from "../components/dashboard/StatCard";
import AlertTable from "../components/dashboard/AlertTable";
import axios from "axios";
import { toast } from "react-hot-toast";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("All");
  const [dateFilter, setDateFilter] = useState("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  const navigate = useNavigate();

const fetchAlerts = async (reset = false) => {
  try {
    setLoading(true);
    setError("");

    const user = JSON.parse(localStorage.getItem("user"));
    const token = localStorage.getItem("authToken");

    if (!token || !user?._id) {
      setError("Missing token or user session. Please login again.");
      return;
    }

    const currentPage = reset ? 1 : page;

    const res = await axios.get(`http://localhost:5000/api/alerts/user/${user._id}`, {
      headers: { Authorization: `Bearer ${token}` },
      params: { page: currentPage, limit: 10 }
    });

    if (res.data.length === 0) {
      setHasMore(false);
    }

    if (reset) {
      setAlerts(res.data);
    } else {
      setAlerts(prev => [...prev, ...res.data]);
    }
  } catch (err) {
    console.error("❌ Error fetching alerts:", err.response || err.message);
    toast.error("Failed to fetch alerts");
    setError("Failed to fetch alerts");
  } finally {
    setLoading(false);
  }
};


  useEffect(() => {
    fetchAlerts();
  }, [page]);

  const handleRefresh = () => {
    fetchAlerts(true); // reset = true
  };

  const todayCount = alerts.filter(
    alert => new Date(alert.timestamp).toDateString() === new Date().toDateString()
  ).length;

  const filteredAlerts = alerts.filter(alert => {
    const matchesType = filter === "All" || alert.type === filter;
    const matchesDate = !dateFilter || new Date(alert.timestamp).toISOString().slice(0, 10) === dateFilter;
    return matchesType && matchesDate;
  });

  return (
    <div className="min-h-screen bg-gray-100 px-6 py-4">
      <Header onRefresh={handleRefresh} />

      <div className="flex gap-4 my-4 flex-wrap">
        <select value={filter} onChange={(e) => setFilter(e.target.value)} className="border p-2 rounded">
          <option value="All">All Types</option>
          <option value="motion_anomaly">Motion</option>
          <option value="vehicle_or_person">Vehicle/Person</option>
        </select>

        <input
          type="date"
          value={dateFilter}
          onChange={(e) => setDateFilter(e.target.value)}
          className="border p-2 rounded"
        />

        <div className="ml-auto flex gap-2">
          <button
            onClick={() => navigate("/rtsp")}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            📡 Start RTSP Stream
          </button>

          <button
            onClick={() => navigate("/upload")}
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
          >
            📤 Upload Video
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        <StatCard title="Total Alerts" value={filteredAlerts.length} icon="📈" color="blue" />
        <StatCard
          title="Motion Anomalies"
          value={filteredAlerts.filter(a => a.type === "motion_anomaly").length}
          icon="⚠️"
          color="orange"
        />
        <StatCard
          title="Vehicle/Person"
          value={filteredAlerts.filter(a => a.type === "vehicle_or_person").length}
          icon="🧍"
          color="green"
        />
        <StatCard title="Today" value={todayCount} icon="📅" color="purple" />
      </div>

      <AlertTable
        alerts={filteredAlerts}
        filter={filter}
        setFilter={setFilter}
        onRefresh={handleRefresh}
        error={error}
      />

      {loading && <p className="text-center my-4 text-gray-500">Loading...</p>}

      {hasMore && !loading && (
        <div className="flex justify-center mt-4">
          <button
            onClick={() => setPage(prev => prev + 1)}
            className="px-4 py-2 bg-blue-600 text-white rounded"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  );
}
