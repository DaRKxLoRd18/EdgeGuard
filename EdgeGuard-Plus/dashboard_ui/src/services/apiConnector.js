import axios from "axios";

export const axiosInstance = axios.create({});

export const apiConnector = (method, url, bodyData = null, headers = {}, params = {}) => {
  return axiosInstance({
    method: method,
    url: url,
    data: bodyData,
    headers: headers,
    params: params,
  });
};