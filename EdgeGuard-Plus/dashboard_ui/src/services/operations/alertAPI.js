import { apiConnector } from "../apiConnector";
import { endpoints } from "../apis";

export const getAllAlerts = async () => {
  try {
    const res = await apiConnector("GET", endpoints.GET_ALERTS_API);
    return res.data;
  } catch (err) {
    console.error("GET_ALERTS_API ERROR:", err);
    return [];
  }
};
