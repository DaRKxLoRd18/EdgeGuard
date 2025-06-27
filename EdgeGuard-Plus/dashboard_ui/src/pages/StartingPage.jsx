import React from "react";
import Login from "./Login";
import GlitchText from "../components/common/GlitchText";
import RotatingText from "../components/common/RotatingText";

const StartingPage = () => {
  return (
    <div className="min-h-screen bg-soft-gradient flex justify-center items-center">
      {/* Floating Window */}
      <div className="relative w-11/12 max-w-[1200px] bg-white flex lg:flex-col backdrop-blur-lg rounded-xl shadow-2xl border border-white/40">
        {/* Mac window buttons */}
        <div className="flex items-center space-x-2 px-4 py-2 border-b-[1px]">
          <span className="w-3 h-3 bg-red-500 rounded-full"></span>
          <span className="w-3 h-3 bg-yellow-400 rounded-full"></span>
          <span className="w-3 h-3 bg-green-500 rounded-full"></span>
        </div>

        <div className=" lg:min-h-[600px]">
          <div className="p-6">
            <div className=" flex  justify-evenly items-center gap-8">
              <div className="flex flex-col items-center text-center">
                <div>
                  <GlitchText
                    speed={0.8}
                    enableShadows={true}
                    enableOnHover={false}
                    className="text-3xl lg:text-5xl xl:text-6xl"
                  >
                    reactbits
                  </GlitchText>
                </div>

                <div>
                  <RotatingText
                    texts={["+", "Plus"]}
                    mainClassName="font-black text-2xl lg:text-4xl xl:text-5xl transition-all duration-500 overflow-hidden w-full justify-center"
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

              <div>
                <Login />
              </div>
            </div>
          </div>
        </div>

        {/* Empty body */}
      </div>
    </div>
  );
};

export default StartingPage;
