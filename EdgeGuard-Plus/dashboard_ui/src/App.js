import "./App.css";
import LoginPage from "./pages/Login";
import {Route, Routes } from "react-router-dom";
import StartingPage from "./pages/StartingPage";


function App() {
  return (
    <div className="w-full min-h-screen bg-richblack-900 flex flex-col font-inter">
        <Routes>
            <Route path="/" element={<StartingPage />}/>
        </Routes>
    </div>
  );
}

export default App;
