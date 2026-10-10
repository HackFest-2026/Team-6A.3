import json
import logging
import math
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


def _read_manifest(model_dir: Path) -> tuple[tuple[str, ...], dict[str, Path], dict[str, float]]:
    manifest_path = model_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required classifier manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or tuple(class_names) != tuple(CLASS_LABELS):
        raise ValueError(f"{manifest_path} must define the supported classes in the expected order.")

    model_files = manifest.get("model_files")
    probability_weights = manifest.get("probability_weights")
    if not isinstance(model_files, dict) or not isinstance(probability_weights, dict):
        raise ValueError(f"{manifest_path} must define model_files and probability_weights.")
    if set(model_files) != {"resnet", "densenet"} or set(probability_weights) != {"resnet50", "densenet121"}:
        raise ValueError(f"{manifest_path} must define ResNet50 and DenseNet121 checkpoints and weights.")

    checkpoints = {
        key: (model_dir.parent / Path(value)).resolve()
        for key, value in model_files.items()
        if isinstance(value, str)
    }
    weights = {
        key: float(value)
        for key, value in probability_weights.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if len(checkpoints) != len(model_files) or len(weights) != len(probability_weights):
        raise ValueError(f"{manifest_path} contains invalid checkpoint paths or probability weights.")
    if any(weight < 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"{manifest_path} probability weights must be non-negative and sum to 1.")
    return tuple(class_names), checkpoints, weights


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


class ClassifierEnsemble:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.device = device
        self.models: list[tuple[nn.Module, float]] = []
        class_names, checkpoints, weights = _read_manifest(model_dir)
        self.class_labels = tuple(CLASS_LABELS[name] for name in class_names)
        constructors = (
            ("resnet", "resnet50", models.resnet50, "fc"),
            ("densenet", "densenet121", models.densenet121, "classifier"),
        )
        for model_key, weight_key, constructor, output_layer in constructors:
            checkpoint_path = checkpoints[model_key]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Required classification checkpoint not found: {checkpoint_path}")
            network = constructor(weights=None)
            layer = getattr(network, output_layer)
            setattr(network, output_layer, nn.Linear(layer.in_features, len(class_names)))
            payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
            network.load_state_dict(_state_dict(payload), strict=True)
            network.to(device).eval()
            self.models.append((network, weights[weight_key]))
            logger.info("Loaded classification checkpoint %s", checkpoint_path)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        probabilities = sum(
            network(tensor).softmax(dim=1)[0] * weight
            for network, weight in self.models
        )
        return {
            label: float(score)
            for label, score in zip(self.class_labels, probabilities.cpu().tolist(), strict=True)
        }
