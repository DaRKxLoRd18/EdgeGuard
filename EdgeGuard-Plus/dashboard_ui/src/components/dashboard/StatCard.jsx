import React from "react";

const colorMap = {
  blue: "bg-blue-100 text-blue-600",
  orange: "bg-orange-100 text-orange-600",
  green: "bg-green-100 text-green-600",
  purple: "bg-purple-100 text-purple-600",
};

export default function StatCard({ title, value, icon, color }) {
  return (
    <div className={`p-4 rounded-lg shadow-md ${colorMap[color]}`}>
      <div className="text-sm font-medium">{title}</div>
      <div className="text-2xl font-bold flex items-center justify-between mt-1">
        {value}
        <span className="text-xl">{icon}</span>
      </div>
    </div>
  );
}
