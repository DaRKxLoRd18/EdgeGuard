// File: src/services/authAPI.js
import { apiConnector } from "../apiConnector";
import { endpoints } from "../apis";

export const registerUser = async ({ name, email, location }) => {
  try {
    const res = await apiConnector("POST", endpoints.REGISTER_USER_API, {
      name,
      email,
      location,
    });
    return res.data;
  } catch (error) {
    console.error("REGISTER_USER_API error:", error);
    throw error;
  }
};

export const getUserByEmail = async (email) => {
  try {
    const res = await apiConnector("GET", endpoints.GET_USER_BY_EMAIL_API, null, {}, { email });
    return res.data;
  } catch (error) {
    console.error("GET_USER_BY_EMAIL_API error:", error);
    throw error;
  }
};
