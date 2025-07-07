import "./App.css";
import { Route, Routes } from "react-router-dom";
import StartingPage from "./pages/StartingPage";
import Dashboard from "./pages/Dashboard";
import RTSPForm from "./pages/RTSPForm"; // ✅ Import the new form page
import { Toaster } from "react-hot-toast";
import Header from "./components/dashboard/Header";
import UploadForm from "./pages/UploadForm";

function App() {
  return (
    <div className="w-full min-h-screen bg-richblack-900 flex flex-col font-inter">
      <Toaster position="top-center" reverseOrder={false} />
      <Routes>
        <Route path="/" element={<StartingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/rtsp" element={<RTSPForm />} /> {/* ✅ New route */}
        <Route path="/upload" element={<UploadForm />} />
        <Route
          path="*"
          element={<div className="text-black text-center text-3xl mt-10">404 - Page Not Found</div>}
        />
      </Routes>
    </div>
  );
}

export default App;
