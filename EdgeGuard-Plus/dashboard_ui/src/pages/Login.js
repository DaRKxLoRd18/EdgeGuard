import React, { useState } from 'react';
import SubmitButton from '../components/common/SubmitButton';
import { SiSpringsecurity } from "react-icons/si";

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <form className="space-y-6 text-black border-l-[1px] border-r-[1px] border-b-[1px] border-[4px] border-t-blue-500 p-10 rounded-lg">
        {/* <div className="h-[2px] w-full bg-gradient-to-r from-blue-500 to-fuchsia-500" /> */}
        <div className=' flex flex-col justify-center items-center gap-3'>
            <div>
                <SiSpringsecurity  size={35}/>
            </div>
            <div className=' flex flex-col justify-center items-center'>
                <p className=' text-2xl font-bold text-gray-700'>Welcome Back</p>
                <p className='text-md text-black'>Sign in to access your dashboard</p>
            </div>
        </div>

      <div>
        <label htmlFor="email" className="block text-sm font-medium mb-1">
          Email address
        </label>
        <input
          type="email"
          id="email"
          className="w-full px-4 py-2 border-[1px] rounded-lg bg-white/20 backdrop-blur placeholder-gray-600 text-black focus:outline-none focus:ring-2 focus:ring-gray"
          placeholder="you@example.com"
        />
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium mb-1">
          Password
        </label>
        <div className="relative">
          <input
            type={showPassword ? 'text' : 'password'}
            id="password"
            className="w-full px-4 border-[1px] py-2 rounded-lg bg-white/20 backdrop-blur placeholder-gray-600 text-black focus:outline-none focus:ring-2 focus:ring-gray"
            placeholder="********"
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-2 text-sm text-gray"
          >
            {showPassword ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>

      <div>

        <button 
            type="submit"
            className="w-full py-2 px-4 rounded-lg text-white font-semibold flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-fuchsia-500 hover:opacity-90 transition"
        >
            Sign Up
        </button>

      </div>
    </form>
  );
}
