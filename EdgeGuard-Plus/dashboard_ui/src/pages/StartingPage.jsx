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
    <div className="relative min-h-screen bg-auth-gradient dark:bg-black transition-colors duration-300 flex justify-center items-center">
      {/* Floating Mac-style card */}
      <div className="w-11/12 max-w-[1200px] animate-float bg-white/80 dark:bg-white/10 backdrop-blur-lg flex lg:flex-col rounded-xl shadow-2xl border border-white/30 transition-colors">
        
        {/* Header with Mac buttons + Theme toggle */}
        <div className="flex items-center justify-between px-4 py-2 border-b-[1px]">
          <div className="flex gap-1">
            <span className="w-3 h-3 bg-red-500 rounded-full"></span>
            <span className="w-3 h-3 bg-yellow-400 rounded-full"></span>
            <span className="w-3 h-3 bg-green-500 rounded-full"></span>
          </div>
          <ThemeTogleBtn darkMode={darkMode} toggleTheme={toggleTheme} />
        </div>

        <div className="lg:min-h-[700px] w-full">
          <div className="p-6">
            <div className="flex flex-wrap lg:flex-nowrap justify-evenly items-center gap-8">
              
              {/* Left: Logo + Ball */}
              <div className="flex flex-col justify-between h-full gap-y-24">
                <div className="flex items-center justify-center text-center">
                  <GlitchText
                    speed={0.8}
                    enableShadows={true}
                    enableOnHover={false}
                    className="text-3xl lg:text-5xl xl:text-6xl text-black dark:text-white"
                  >
                    EdgeGuard
                  </GlitchText>
                  <RotatingText
                    texts={["+", "Plus"]}
                    mainClassName="font-black text-2xl lg:text-4xl xl:text-5xl transition-all duration-500 overflow-hidden justify-center dark:text-white"
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

                <div className="flex justify-center">
                  <BallEffect />
                </div>
              </div>

              {/* Right: AuthForm (no floating here) */}
              <div>
                <AuthForm />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StartingPage;
