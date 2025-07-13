import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

const BallEffect = () => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const dark = document.documentElement.classList.contains('dark');
    setIsDark(dark);
  }, []);

  return (
    <StyledWrapper ballColor={isDark ? '#60A5F9' : '#2564EA'}>
      <div className="container">
        <div className="ball-container">
          <div className="ball">
            <div className="inner">
              <div className="line" />
              <div className="line line--two" />
              <div className="oval" />
              <div className="oval oval--two" />
            </div>
          </div>
          <div className="shadow" />
        </div>
      </div>
    </StyledWrapper>
  );
};

export default BallEffect;

const StyledWrapper = styled.div`
  .container {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .ball-container {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    width: 60px;
  }

  @keyframes rotateBall {
    0% { transform: rotateY(0deg) rotateX(0deg) rotateZ(0deg); }
    50% { transform: rotateY(360deg) rotateX(360deg) rotateZ(0deg); }
    100% { transform: rotateY(720deg) rotateX(720deg) rotateZ(360deg); }
  }

  @keyframes bounceBall {
    0% { transform: translateY(-70px) scale(1, 1); }
    15% { transform: translateY(-56px) scale(1, 1); }
    45% { transform: translateY(70px) scale(1, 1); }
    50% { transform: translateY(73.5px) scale(1, 0.92); }
    55% { transform: translateY(70px) scale(1, 0.95); }
    85% { transform: translateY(-56px) scale(1, 1); }
    95% { transform: translateY(-70px) scale(1, 1); }
    100% { transform: translateY(-70px) scale(1, 1); }
  }

  .ball {
    animation: bounceBall 1.2s infinite cubic-bezier(0.42, 0, 0.58, 1);
    border-radius: 50%;
    height: 50px;
    width: 50px;
    transform-style: preserve-3d;
    position: relative;
    transform: translateY(-70px);
    z-index: 1;
  }

  .ball::before {
    background: radial-gradient(circle at 36px 20px, ${(props) => props.ballColor}, #1e40af);
    border: 2px solid #333333;
    border-radius: 50%;
    content: "";
    height: calc(100% + 6px);
    width: calc(100% + 6px);
    left: -3px;
    top: -3px;
    position: absolute;
    transform: translateZ(1vmin);
  }

  .ball .inner {
    animation: rotateBall 25s linear infinite;
    border-radius: 50%;
    height: 100%;
    width: 100%;
    position: absolute;
    transform-style: preserve-3d;
  }

  .ball .line::before,
  .ball .line::after {
    border: 2px solid #333333;
    border-radius: 50%;
    content: "";
    height: 99%;
    width: 99%;
    position: absolute;
  }

  .ball .line::before {
    transform: rotate3d(0, 0, 0, 0);
  }

  .ball .line::after {
    transform: rotate3d(1, 0, 0, 90deg);
  }

  .ball .line--two::before {
    transform: rotate3d(0, 0, 0, 2deg);
  }

  .ball .line--two::after {
    transform: rotate3d(1, 0, 0, 88deg);
  }

  .ball .oval::before,
  .ball .oval::after {
    border-top: 4px solid #333333;
    border-radius: 50%;
    content: "";
    height: 99%;
    width: 99%;
    position: absolute;
  }

  .ball .oval::before {
    transform: rotate3d(1, 0, 0, 45deg) translate3d(0, 0, 6px);
  }

  .ball .oval::after {
    transform: rotate3d(1, 0, 0, -45deg) translate3d(0, 0, -6px);
  }

  .ball .oval--two::before {
    transform: rotate3d(1, 0, 0, 135deg) translate3d(0, 0, -6px);
  }

  .ball .oval--two::after {
    transform: rotate3d(1, 0, 0, -135deg) translate3d(0, 0, 6px);
  }

  @keyframes bounceShadow {
    0% { filter: blur(3px); opacity: 0.6; transform: translateY(73px) scale(0.5, 0.5); }
    45% { filter: blur(1px); opacity: 0.9; transform: translateY(73px) scale(1, 1); }
    55% { filter: blur(1px); opacity: 0.9; transform: translateY(73px) scale(1, 1); }
    100% { filter: blur(3px); opacity: 0.6; transform: translateY(73px) scale(0.5, 0.5); }
  }

  .shadow {
    animation: bounceShadow 1.2s infinite cubic-bezier(0.42, 0, 0.58, 1);
    background: black;
    border-radius: 50%;
    filter: blur(2px);
    height: 6px;
    width: 54px;
    align-self: center;
  }
`;
