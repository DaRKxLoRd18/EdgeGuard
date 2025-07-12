import React from "react";

export default function AlertTable({ alerts, filter, setFilter, onRefresh, error }) {
  const filtered = filter === "All" ? alerts : alerts.filter(a => a.type === filter.toLowerCase());

  return (
    <div className="mt-8 bg-white dark:bg-gray-800 p-4 rounded shadow text-gray-800 dark:text-gray-200">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-xl font-semibold">Recent Alerts</h2>
        <div className="flex gap-3 items-center">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border px-2 py-1 rounded dark:bg-gray-700 dark:text-white"
          >
            <option>All</option>
            <option>Motion</option>
            <option>Object</option>
          </select>
          <button
            onClick={onRefresh}
            className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition"
          >
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="bg-red-100 dark:bg-red-800 text-red-600 dark:text-red-300 px-4 py-2 rounded">
          {`Error: ${error}`}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center text-gray-500 dark:text-gray-400 py-6">No alerts available</div>
      ) : (
        <table className="w-full text-sm border-t dark:border-gray-600">
          <thead>
            <tr className="text-left border-b dark:border-gray-600">
              <th className="p-2">Type</th>
              <th className="p-2">Time</th>
              <th className="p-2">Location</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((alert, idx) => (
              <tr
                key={idx}
                className="border-b dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
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
