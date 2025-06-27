import { useEffect, useState } from "react";
import "./GlitchText.css"; // 🔹 Make sure this file exists with glitch styles

const GlitchText = ({
  children,
  speed = 1,
  enableShadows = true,
  enableOnHover = true,
  className = "",
}) => {
  const [isAnimating, setIsAnimating] = useState(!enableOnHover);

  useEffect(() => {
    if (!enableOnHover) {
      setIsAnimating(true);
    }
  }, [enableOnHover]);

  const handleMouseEnter = () => {
    if (enableOnHover) setIsAnimating(true);
  };

  const handleMouseLeave = () => {
    if (enableOnHover) setIsAnimating(false);
  };

  return (
    <div
      className={`glitch-text ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      data-text={children}
    >
      {children}

      {isAnimating && (
        <>
          <span
            className="glitch-layer glitch-layer-after"
            style={{
              animation: `glitch-anim ${speed * 3}s infinite linear alternate-reverse`,
              textShadow: enableShadows ? "-5px 0 red" : "none",
            }}
          >
            {children}
          </span>
          <span
            className="glitch-layer glitch-layer-before"
            style={{
              animation: `glitch-anim ${speed * 2}s infinite linear alternate-reverse`,
              textShadow: enableShadows ? "5px 0 cyan" : "none",
            }}
          >
            {children}
          </span>
        </>
      )}
    </div>
  );
};

export default GlitchText;
