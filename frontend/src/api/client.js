import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function predictSignal(signal, samplingRate = 250) {
  const payload = {
    signal,
    sampling_rate: samplingRate,
  };
  const { data } = await apiClient.post("/predict", payload);
  return data;
}

export async function uploadSignalFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await apiClient.post("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function generateReport(patientId, prediction, notes = "") {
  const payload = {
    patient_id: patientId,
    prediction,
    notes,
  };
  const { data } = await apiClient.post("/report", payload);
  return data;
}

