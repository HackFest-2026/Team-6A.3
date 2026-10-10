from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut, convert_color_space

from app.config import settings
from app.schemas import PatientDetails


def decode_image(data: bytes, filename: str) -> Image.Image:
    return decode_upload(data, filename)[0]


def _patient_details(dataset: pydicom.Dataset) -> PatientDetails:
    def value(tag: str, limit: int = 128) -> str | None:
        raw_value = getattr(dataset, tag, None)
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        return text[:limit] or None

    return PatientDetails(
        patient_name=value("PatientName"),
        patient_id=value("PatientID"),
        patient_age=value("PatientAge", 16),
        patient_sex=value("PatientSex", 16),
        study_date=value("StudyDate", 16),
    )


def _to_uint8(pixels: np.ndarray) -> np.ndarray:
    if pixels.dtype == np.uint8:
        return pixels
    pixels = pixels.astype(np.float32)
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise ValueError("DICOM pixel data contains no finite values.")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        high = low + 1
    return np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _decode_dicom(data: bytes) -> tuple[Image.Image, PatientDetails]:
    try:
        dataset = pydicom.dcmread(BytesIO(data))
        pixels = np.asarray(dataset.pixel_array)
        samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
        if samples_per_pixel == 1 and pixels.ndim == 3:
            pixels = pixels[0]
        elif samples_per_pixel > 1 and pixels.ndim == 4:
            pixels = pixels[0]

        photometric = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
        if samples_per_pixel == 1:
            pixels = apply_modality_lut(pixels, dataset)
            if getattr(dataset, "VOILUTSequence", None) or hasattr(dataset, "WindowCenter"):
                pixels = apply_voi_lut(pixels, dataset)
            pixels = _to_uint8(pixels)
            if photometric == "MONOCHROME1":
                pixels = 255 - pixels
            image = Image.fromarray(pixels, mode="L").convert("RGB")
        else:
            if photometric.startswith("YBR"):
                pixels = convert_color_space(pixels, photometric, "RGB")
            pixels = _to_uint8(pixels)
            image = Image.fromarray(pixels).convert("RGB")
        if image.width == 0 or image.height == 0:
            raise ValueError("DICOM image has invalid dimensions.")
        return image, _patient_details(dataset)
    except Exception as error:
        raise HTTPException(status_code=400, detail="The DICOM file is invalid, unsupported, or has unreadable pixel data.") from error


def decode_upload(data: bytes, filename: str) -> tuple[Image.Image, PatientDetails | None]:
    extension = Path(filename).suffix.lower() if filename else ""
    if extension not in settings.allowed_extensions:
        raise HTTPException(status_code=415, detail="Upload a JPG, JPEG, PNG, or DICOM image.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MB upload limit.")
    if extension in (".dcm", ".dicom"):
        image, patient_details = _decode_dicom(data)
        return image, patient_details
    try:
        with Image.open(BytesIO(data)) as image:
            expected_format = "PNG" if extension == ".png" else "JPEG"
            if image.format != expected_format:
                raise HTTPException(status_code=415, detail="Image contents do not match the file extension.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="The image is corrupted or unreadable.") from error


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MB upload limit.")
    return data
