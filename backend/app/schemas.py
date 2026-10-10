from datetime import datetime
from pydantic import BaseModel


class PatientDetails(BaseModel):
    patient_name: str | None = None
    patient_id: str | None = None
    patient_age: str | None = None
    patient_sex: str | None = None
    study_date: str | None = None


class PredictionResponse(BaseModel):
    id: str
    filename: str
    predicted_class: str
    probabilities: dict[str, float]
    yolo_confidence: float | None
    original_image: str
    annotated_image: str
    patient_details: PatientDetails | None = None
    created_at: datetime
    inference_ms: float


class HistoryItem(BaseModel):
    id: str
    filename: str
    predicted_class: str
    confidence: float
    created_at: datetime
    inference_ms: float
