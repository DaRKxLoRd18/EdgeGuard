import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-hot-toast";
import Header from "../components/dashboard/Header";

export default function UploadForm() {
  const [videoFile, setVideoFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile) return toast.error("Please select a video file");

    const user = JSON.parse(localStorage.getItem("user"));
    if (!user?.email) return toast.error("No user email found");

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("email", user.email);

    try {
      const res = await axios.post("http://localhost:5000/api/upload", formData);
      toast.success(res.data.message);
    } catch (err) {
      toast.error(err.response?.data?.message || "Upload failed");
    }
  };

  return (
    <div>
        <div>
            <Header/>
        </div>
        <div className=" w-full flex justify-center items-center min-h-screen">
             <div className="p-6 max-w-xl mx-auto bg-white rounded shadow flex flex-col     justify-center items-center">
                <h2 className="text-xl font-bold mb-4">Upload Video for Anomaly Detection</h2>
                <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                    <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => setVideoFile(e.target.files[0])}
                    className="border p-2 rounded"
                    />
                    <button
                    type="submit"
                    className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
                    >
                    Upload & Process
                    </button>
                </form>
            </div>
        </div>
       
    </div>
    
  );
}
