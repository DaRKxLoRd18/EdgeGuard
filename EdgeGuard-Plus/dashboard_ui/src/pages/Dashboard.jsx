import React, { useEffect, useState } from "react";
import Header from "../components/dashboard/Header";
import StatCard from "../components/dashboard/StatCard";
import AlertTable from "../components/dashboard/AlertTable";
import axios from "axios";

export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("All");

  const fetchAlerts = async () => {
    try {
      setError("");
      const user = JSON.parse(localStorage.getItem("user"));
      const res = await axios.get("http://localhost:5000/api/alerts", {
        params: { userId: user._id },
      });
      setAlerts(res.data);
    } catch (err) {
      setError("Failed to fetch alerts");
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const todayCount = alerts.filter(alert =>
    new Date(alert.timestamp).toDateString() === new Date().toDateString()
  ).length;

  return (
    <div className="min-h-screen bg-gray-100 px-6 py-4">
      <Header onRefresh={fetchAlerts} />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        <StatCard title="Total Alerts" value={alerts.length} icon="📈" color="blue" />
        <StatCard title="Motion Anomalies" value={alerts.filter(a => a.type === "motion").length} icon="⚠️" color="orange" />
        <StatCard title="Vehicle/Person" value={alerts.filter(a => a.type === "object").length} icon="🧍" color="green" />
        <StatCard title="Today" value={todayCount} icon="📅" color="purple" />
      </div>

      <AlertTable
        alerts={alerts}
        filter={filter}
        setFilter={setFilter}
        onRefresh={fetchAlerts}
        error={error}
      />
    </div>
  );
}
