import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

logger = logging.getLogger(__name__)
MANIFEST_NAME = "AS4_FINAL_ensemble.json"
CLASS_LABELS = {
    "Normal": "Healthy",
    "No Lung Opacity / Not Normal": "Not Healthy",
    "Lung Opacity": "Pneumonia",
}


def _read_manifest(model_dir: Path) -> tuple[tuple[str, ...], Path]:
    manifest_path = model_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required classifier manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or tuple(class_names) != tuple(CLASS_LABELS):
        raise ValueError(f"{manifest_path} must define the supported classes in the expected order.")

    model_files = manifest.get("model_files")
    if not isinstance(model_files, dict) or set(model_files) != {"densenet"}:
        raise ValueError(f"{manifest_path} must define only the DenseNet121 checkpoint.")
    checkpoint_file = model_files["densenet"]
    if not isinstance(checkpoint_file, str):
        raise ValueError(f"{manifest_path} contains an invalid DenseNet121 checkpoint path.")
    return tuple(class_names), (model_dir.parent / Path(checkpoint_file)).resolve()


def _state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")
    return {key.removeprefix("module."): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


class DenseNet121Classifier:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.device = device
        class_names, checkpoint_path = _read_manifest(model_dir)
        self.class_labels = tuple(CLASS_LABELS[name] for name in class_names)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required DenseNet121 classification checkpoint not found: {checkpoint_path}")
        self.model = models.densenet121(weights=None)
        layer = self.model.classifier
        self.model.classifier = nn.Linear(layer.in_features, len(class_names))
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(_state_dict(payload), strict=True)
        self.model.to(device).eval()
        logger.info("Loaded DenseNet121 classification checkpoint %s", checkpoint_path)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        probabilities = self.model(tensor).softmax(dim=1)[0]
        return {
            label: float(score)
            for label, score in zip(self.class_labels, probabilities.cpu().tolist(), strict=True)
        }
