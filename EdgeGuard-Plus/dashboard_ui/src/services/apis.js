const BASE_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const endpoints = {
  REGISTER_USER_API: BASE_URL + "/api/users/register",
  LOGIN_USER_API: BASE_URL + "/api/users/login", // If implemented
  GET_USER_BY_EMAIL_API: BASE_URL + "/api/users/by-email",
  GET_USER_HISTORY_API: BASE_URL + "/api/users/history", // or /:id
  POST_ALERT_API: BASE_URL + "/api/alerts",
  GET_ALERTS_API: BASE_URL + "/api/alerts",
  GET_ANOMALIES_API: BASE_URL + "/api/anomalies",
};