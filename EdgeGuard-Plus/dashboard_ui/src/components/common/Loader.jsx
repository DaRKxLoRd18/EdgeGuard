// import React from "react";

// export default function Loader() {
//   return (
//     <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100 text-center p-4">
//       <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 border-solid mb-6"></div>
//       <p className="text-lg font-semibold text-gray-700">
//         Video is analyzing... Please wait or go back to Dashboard.
//       </p>
//     </div>
//   );
// }


import React from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';

const Loader = () => {
  const navigate = useNavigate();

  return (
    <StyledWrapper>
      <div className="cube">
        <div className="top" />
        <div className="sides">
          {[0, 1, 2, 3].map(i => (
            <span className="face" style={{ '--i': i }} key={i}>
              <p>Analyzing...</p>
              <p>Analyzing...</p>
            </span>
          ))}
        </div>
      </div>

      <p className="message">🎥 Video is analyzing... Please wait or go back to Dashboard.</p>

      <button className="go-button " onClick={() => navigate('/dashboard')}>
        Go to Dashboard
      </button>
    </StyledWrapper>
  );
};

export default Loader;



const StyledWrapper = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: #f7fafc;

  .cube {
    position: relative;
    width: 100px;
    height: 100px;
    transform-style: preserve-3d;
    animation: spin 4s linear infinite;
  }

  @keyframes spin {
    0% {
      transform: rotateX(-30deg) rotateY(0deg);
    }
    100% {
      transform: rotateX(-30deg) rotateY(360deg);
    }
  }

  .sides {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    transform-style: preserve-3d;
  }

  .face {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    transform: rotateY(calc(90deg * var(--i))) translateZ(50px);
    background: linear-gradient(-45deg, #0f172a, #334155, #1e293b);
    background-size: 1200% 1200%;
    animation: faceAnim 30s ease infinite;
    display: flex;
    justify-content: center;
    align-items: center;
    transform-style: preserve-3d;
  }

  .face p {
    position: absolute;
    font-size: 0.9rem;
    color: #fff;
    transform: translateZ(20px);
    font-weight: bold;
  }

  .face p:nth-child(1) {
    transform: translateZ(0) translateY(20px);
    color: rgba(255, 255, 255, 0.1);
    filter: blur(2px);
  }

  .top {
    position: absolute;
    top: 0;
    left: 0;
    width: 100px;
    height: 100px;
    background: linear-gradient(-45deg, #1e1b4b, #0f172a);
    transform: rotateX(90deg) translateZ(50px);
    background-size: 1200% 1200%;
    animation: faceAnim 30s ease infinite;
  }

  @keyframes faceAnim {
    0% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
    100% {
      background-position: 0% 50%;
    }
  }

  .message {
    margin-top: 30px;
    font-size: 1.1rem;
    color: #1f2937;
    font-weight: 500;
    text-align: center;
  }
`;
