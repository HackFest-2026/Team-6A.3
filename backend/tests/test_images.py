from io import BytesIO

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.config import settings
from app.images import decode_image, decode_upload


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (8, 8), color=180).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(180, 180, 180)).save(buffer, format="JPEG")
    return buffer.getvalue()


def dicom_bytes() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Example^Patient"
    dataset.PatientID = "P-1024"
    dataset.PatientAge = "042Y"
    dataset.PatientSex = "F"
    dataset.StudyDate = "20260614"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.array([[0, 100, 200], [300, 400, 500]], dtype="<u2").tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_decode_image_returns_rgb() -> None:
    image = decode_image(png_bytes(), "scan.PNG")
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_decode_dicom_converts_pixels_and_extracts_patient_details() -> None:
    image, details = decode_upload(dicom_bytes(), "study.dcm")

    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert details is not None
    assert details.patient_name == "Example^Patient"
    assert details.patient_id == "P-1024"
    assert details.patient_age == "042Y"
    assert details.patient_sex == "F"
    assert details.study_date == "20260614"


def test_decode_dicom_rejects_unreadable_file() -> None:
    with pytest.raises(HTTPException) as error:
        decode_upload(b"not a DICOM file", "study.dcm")
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    ("content", "filename", "status"),
    [
        (b"", "scan.png", 400),
        (b"not an image", "scan.png", 400),
        (png_bytes(), "scan.gif", 415),
        (jpeg_bytes(), "scan.png", 415),
    ],
)
def test_decode_image_rejects_invalid_uploads(content: bytes, filename: str, status: int) -> None:
    with pytest.raises(HTTPException) as error:
        decode_image(content, filename)
    assert error.value.status_code == status


def test_decode_image_rejects_oversized_upload() -> None:
    with pytest.raises(HTTPException) as error:
        decode_image(b"x" * (settings.max_upload_bytes + 1), "scan.png")
    assert error.value.status_code == 413
