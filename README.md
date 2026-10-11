# Radiant Pneumonia Detection

A full-stack chest X-ray analysis workspace. DenseNet121 provides the three-class disease classification; YOLO11s only localizes suspicious regions and reports its highest detection confidence. DenseNet121 remains the sole authority for the predicted class.

> This software is an AI-assisted screening tool, not a standalone diagnosis. All outputs require review by a qualified clinician.

## Requirements

- Node.js 20 or newer and npm
- Python 3.10 or newer
- Classifier manifest and checkpoint: `AS4_FINAL_ensemble.json` and `AS4_FINAL_ensemble_densenet121.pth`
- YOLO localization checkpoint: `controlled_v1_best.pt` or `yolo.pth`

## Model checkpoints

Put the classifier manifest and DenseNet121 checkpoint, plus `controlled_v1_best.pt` or `yolo.pth`, in `backend/models/`. The API reads the classifier filename and class order from `AS4_FINAL_ensemble.json`; its source classes are mapped to `Healthy`, `Not Healthy`, and `Pneumonia` in that order. See `backend/models/README.md` for checkpoint formats. The API loads DenseNet121 and the localizer at startup and exits with an explicit error if required files are missing or incompatible. It does not produce placeholder predictions.

The disease classes are ordered `Healthy`, `Not Healthy`, `Pneumonia`. The DenseNet121 checkpoint must have a three-logit output head. The YOLO checkpoint must be compatible with Ultralytics YOLO.

## Run locally

From the project root, run:

```powershell
 .\start.ps1
```

On first launch, the script creates `.venv` if needed, installs missing backend and frontend dependencies, starts the API, waits for its model readiness check, and then starts Vite. Keep the API console open while using the app; press Ctrl+C in the Vite terminal to stop the local services. Python 3.10 or newer and Node.js 20 or newer must be installed. If PowerShell blocks local scripts, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then run the script again.

Open the Vite URL printed by npm. In local development, API requests are proxied through Vite to `http://127.0.0.1:8000`, avoiding browser host and CORS mismatches. FastAPI docs are available at `http://localhost:8000/docs` after model loading succeeds. Set `VITE_API_URL` to use a different API origin.

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
