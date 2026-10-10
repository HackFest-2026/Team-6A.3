# Radiant Pneumonia Detection

A full-stack chest X-ray analysis workspace. ResNet50 and DenseNet121 provide the three-class disease classification ensemble; YOLO11s only localizes suspicious regions and reports its highest detection confidence. The classifier remains the sole authority for the predicted class.

> This software is an AI-assisted screening tool, not a standalone diagnosis. All outputs require review by a qualified clinician.

## Requirements

- Node.js 20 or newer and npm
- Python 3.10 or newer
- Classifier manifest and checkpoints: `AS4_FINAL_ensemble.json`, `AS4_FINAL_ensemble_resnet50.pth`, and `AS4_FINAL_ensemble_densenet121.pth`
- YOLO localization checkpoint: `controlled_v1_best.pt` or `yolo.pth`

## Model checkpoints

Put the classifier manifest and checkpoints, plus `controlled_v1_best.pt` or `yolo.pth`, in `backend/models/`. The API reads classifier filenames, class order, and ensemble weights from `AS4_FINAL_ensemble.json`; its source classes are mapped to `Healthy`, `Not Healthy`, and `Pneumonia` in that order. See `backend/models/README.md` for checkpoint formats. The API loads the classifiers and localizer at startup and exits with an explicit error if required files are missing or incompatible. It does not produce placeholder predictions.

The disease classes are ordered `Healthy`, `Not Healthy`, `Pneumonia`. The classifier checkpoints must have a three-logit output head. The YOLO checkpoint must be compatible with Ultralytics YOLO.

## Run locally

Install frontend dependencies:

```powershell
cd frontend
npm install
npm run dev
```

In a second terminal, install backend dependencies and start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Open the Vite URL printed by npm, normally `http://localhost:5173`. The frontend expects the API at `http://localhost:8000`; set `VITE_API_URL` before starting Vite to override it. FastAPI docs are available at `http://localhost:8000/docs` after model loading succeeds.

For PyTorch GPU acceleration, install the PyTorch build matching the machine's CUDA setup before installing the remaining requirements. CPU inference is selected automatically when CUDA is unavailable.

## API

- `POST /predict`: multipart field `file`; returns classifier probabilities, predicted class, original and annotated PNGs (base64), YOLO maximum confidence (null when no boxes), and timing.
- `GET /history`: in-memory metadata for recent uploads (up to 20).
- `DELETE /history`: clears runtime history.
- `GET /health`: readiness check.

Uploads may be valid JPG/JPEG/PNG images or DICOM (`.dcm` / `.dicom`) studies no larger than 5 MB. DICOM pixel data is decoded from the upload buffer into RGB for the existing model pipeline. Available patient name, ID, age, sex, and study date are returned with that prediction and are not added to runtime history or written to disk. History is held in memory only and resets when the API process restarts.

## Tests

From the workspace root, run:

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests -q
```

The current focused tests cover image decoding and upload validation. Full model inference requires the trained checkpoints.
