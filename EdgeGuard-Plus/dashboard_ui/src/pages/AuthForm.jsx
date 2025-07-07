import React, { useState } from 'react';
import { SiSpringsecurity } from "react-icons/si";
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

export default function AuthForm() {
  const navigate = useNavigate();
  const [mode, setMode] = useState("login");
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    location: ""
  });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const toggleMode = () => {
    setMode(prev => (prev === "login" ? "signup" : "login"));
    setFormData({ name: "", email: "", password: "", location: "" });
    setError("");
  };

  const handleChange = (e) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (!formData.email || !formData.password) {
        setError("Email and password are required.");
        setLoading(false);
        return;
      }

      if (mode === "signup") {
        if (!formData.name || !formData.location) {
          setError("All fields are required for signup.");
          setLoading(false);
          return;
        }

        const res = await axios.post("http://localhost:5000/api/users/register", {
          name: formData.name,
          email: formData.email,
          password: formData.password,
          location: formData.location
        });

        toast.success("✅ Registration Successful!");
        setMode("login");
        setFormData({ name: "", email: "", password: "", location: "" });

      } else {
        const res = await axios.post("http://localhost:5000/api/users/login", {
          email: formData.email,
          password: formData.password
        });

        toast.success("✅ Login Successful!");

        if (res.data.token) {
          localStorage.setItem("authToken", res.data.token);
          localStorage.setItem("user", JSON.stringify({
            _id: res.data._id,
            name: res.data.name,
            email: res.data.email
          }));
        }

        navigate("/dashboard");
      }
    } catch (err) {
      toast.error(err.response?.data?.message || "❌ Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-6 text-black border border-blue-500 p-10 rounded-lg w-[300px] sm:w-[350px]"
    >
      <div className="flex flex-col justify-center items-center gap-3">
        <SiSpringsecurity size={35} />
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-700">
            {mode === "login" ? "Welcome Back" : "Create Account"}
          </p>
          <p className="text-md text-black">
            {mode === "login"
              ? "Sign in to access your dashboard"
              : "Register to get started"}
          </p>
        </div>
      </div>

      {error && <p className="text-red-500 text-center">{error}</p>}

      {mode === "signup" && (
        <>
          <div>
            <label htmlFor="name" className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              name="name"
              required
              value={formData.name}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-lg bg-white/20 placeholder-gray-600"
              placeholder="Your name"
            />
          </div>

          <div>
            <label htmlFor="location" className="block text-sm font-medium mb-1">Location</label>
            <input
              type="text"
              name="location"
              required
              value={formData.location}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-lg bg-white/20 placeholder-gray-600"
              placeholder="Your location"
            />
          </div>
        </>
      )}

      <div>
        <label htmlFor="email" className="block text-sm font-medium mb-1">Email address</label>
        <input
          type="email"
          name="email"
          required
          value={formData.email}
          onChange={handleChange}
          className="w-full px-4 py-2 border rounded-lg bg-white/20 placeholder-gray-600"
          placeholder="you@example.com"
        />
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium mb-1">Password</label>
        <div className="relative">
          <input
            type={showPassword ? 'text' : 'password'}
            name="password"
            required
            value={formData.password}
            onChange={handleChange}
            className="w-full px-4 py-2 border rounded-lg bg-white/20 placeholder-gray-600"
            placeholder="********"
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-2 text-sm"
          >
            {showPassword ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className={`w-full py-2 px-4 text-white font-semibold rounded-lg transition ${
          loading
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-gradient-to-r from-blue-500 to-fuchsia-500 hover:opacity-90"
        }`}
      >
        {loading ? "Processing..." : mode === "login" ? "Login" : "Sign Up"}
      </button>

      <p className="text-center text-sm">
        {mode === "login" ? "Don't have an account?" : "Already have an account?"}{" "}
        <button type="button" onClick={toggleMode} className="text-blue-600 underline">
          {mode === "login" ? "Sign Up" : "Login"}
        </button>
      </p>
    </form>
  );
}
