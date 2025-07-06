import React from "react";

export default function AlertTable({ alerts, filter, setFilter, onRefresh, error }) {
  const filtered = filter === "All" ? alerts : alerts.filter(a => a.type === filter.toLowerCase());

  return (
    <div className="mt-8 bg-white p-4 rounded shadow">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-xl font-semibold">Recent Alerts</h2>
        <div className="flex gap-3 items-center">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border px-2 py-1 rounded"
          >
            <option>All</option>
            <option>Motion</option>
            <option>Object</option>
          </select>
          <button onClick={onRefresh} className="bg-blue-600 text-white px-3 py-1 rounded">
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="bg-red-100 text-red-600 px-4 py-2 rounded">{`Error: ${error}`}</div>
      ) : filtered.length === 0 ? (
        <div className="text-center text-gray-500 py-6">No alerts available</div>
      ) : (
        <table className="w-full text-sm border-t">
          <thead>
            <tr className="text-left border-b">
              <th className="p-2">Type</th>
              <th className="p-2">Time</th>
              <th className="p-2">Location</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((alert, idx) => (
              <tr key={idx} className="border-b hover:bg-gray-50">
                <td className="p-2 capitalize">{alert.type}</td>
                <td className="p-2">{new Date(alert.timestamp).toLocaleString()}</td>
                <td className="p-2">{alert.location}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
