import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import GlitchText from "../components/common/GlitchText";
import RotatingText from "../components/common/RotatingText";
import AuthForm from "./AuthForm";
import ThemeTogleBtn from "../components/common/ThemeTogleBtn";
import BallEffect from "../components/common/BallEffect";

const StartingPage = () => {
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("theme") === "dark");

  useEffect(() => {
    const user = localStorage.getItem("user");
    if (user) navigate("/dashboard");
  }, [navigate]);

  useEffect(() => {
    if (darkMode) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const toggleTheme = () => setDarkMode((prev) => !prev);

  return (
    <div className="relative min-h-screen transition-colors duration-300 flex justify-center items-center px-4 pt-8 pb-8 bg-[#EADFD1] dark:bg-[#212A3B]">
      {/* Floating Mac-style card with horizontal gradient */}
      <div
        className={`w-full max-w-[1100px] min-h-[600px] animate-float 
          rounded-xl shadow-2xl border border-white/30 transition-colors flex flex-col 
          backdrop-blur-lg overflow-hidden ${
            darkMode ? "bg-gradient-dark" : "bg-gradient-light"
          }`}
      >
        {/* Header with Mac buttons + Theme toggle */}
        <div className="flex items-center justify-between px-4 py-2 border-b-[1px]">
          <div className="flex gap-1">
            <span className="w-3 h-3 bg-red-500 rounded-full cursor-pointer"></span>
            <span className="w-3 h-3 bg-yellow-400 rounded-full cursor-pointer"></span>
            <span className="w-3 h-3 bg-green-500 rounded-full cursor-pointer"></span>
          </div>
          <ThemeTogleBtn darkMode={darkMode} toggleTheme={toggleTheme} />
        </div>

        {/* Centered Content */}
        <div className="flex flex-1 justify-center items-center px-6 py-6">
          <div className="flex flex-col lg:flex-row justify-center items-center gap-16 w-full">
            {/* Left: Title + BallEffect */}
            <div className="flex flex-col items-center justify-center gap-6">
              <div className="flex items-center gap-x-0 text-center">
                <GlitchText
                  speed={0.8}
                  enableShadows={true}
                  enableOnHover={false}
                  className="text-4xl lg:text-5xl text-[#374151] dark:text-[#D1E8EC] font-black mr-1"
                >
                  EdgeGuard
                </GlitchText>
                <div className="w-[110px]">
                  <RotatingText
                    texts={["+", "Plus"]}
                    mainClassName="font-black text-4xl lg:text-5xl text-[#2564EA] dark:text-[#60A5F9] transition-all duration-500 overflow-hidden justify-center"
                    staggerFrom="last"
                    initial={{ y: "100%" }}
                    animate={{ y: 0 }}
                    exit={{ y: "-120%" }}
                    staggerDuration={0.05}
                    splitLevelClassName="overflow-hidden"
                    transition={{ type: "spring", damping: 30, stiffness: 400 }}
                    rotationInterval={3000}
                    splitBy="characters"
                  />
                </div>
              </div>

              {/* Ball Effect */}
              <div className="mt-12">
                <BallEffect />
              </div>
            </div>

            {/* Right: Auth Form */}
            <div className="w-full max-w-[350px]">
              <AuthForm />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StartingPage;
