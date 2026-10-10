import asyncio
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi import UploadFile
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app import main


def dicom_bytes() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Example^Patient"
    dataset.PatientID = "P-2048"
    dataset.PatientAge = "052Y"
    dataset.PatientSex = "F"
    dataset.StudyDate = "20261010"
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(20, dtype="<u2").reshape(4, 5).tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_dicom_upload_flows_through_models_and_returns_prediction() -> None:
    class FakeClassifier:
        received_image = None

        def predict(self, image):
            self.received_image = image
            return {"Healthy": 0.8, "Not Healthy": 0.1, "Pneumonia": 0.1}

    class FakeDetector:
        received_image = None

        def annotate(self, image):
            self.received_image = image
            return image.copy(), 0.72

    classifier = FakeClassifier()
    detector = FakeDetector()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(classifier=classifier, detector=detector))
    )
    upload = UploadFile(filename="study.dcm", file=BytesIO(dicom_bytes()))
    main.history.clear()

    result = asyncio.run(main.predict(request, upload))

    assert classifier.received_image.mode == "RGB"
    assert classifier.received_image.size == (5, 4)
    assert detector.received_image.shape == (4, 5, 3)
    assert result.predicted_class == "Healthy"
    assert result.yolo_confidence == 0.72
    assert result.patient_details.patient_id == "P-2048"
    assert result.patient_details.patient_age == "052Y"
    assert result.original_image
    assert result.annotated_image
    assert len(main.history) == 1
    assert not hasattr(main.history[0], "patient_id")
    main.history.clear()