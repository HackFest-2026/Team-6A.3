import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api',
  timeout: 120_000,
})

export type PatientDetails = {
  patient_name: string | null
  patient_id: string | null
  patient_age: string | null
  patient_sex: string | null
  study_date: string | null
}

export type Prediction = {
  id: string
  filename: string
  predicted_class: string
  probabilities: Record<string, number>
  yolo_confidence: number | null
  original_image: string
  annotated_image: string
  patient_details: PatientDetails | null
  created_at: string
  inference_ms: number
}

export type HistoryItem = {
  id: string
  filename: string
  predicted_class: string
  confidence: number
  created_at: string
  inference_ms: number
}
