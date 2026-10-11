import json

import pytest
import torch
from PIL import Image
from torch import nn

import app.predict as prediction_module


class FakeNetwork(nn.Module):
    def __init__(self, output_layer: str, logits: list[float]) -> None:
        super().__init__()
        setattr(self, output_layer, nn.Linear(1, 3))
        self.logits = logits

    def load_state_dict(self, _state_dict: dict, **_kwargs) -> None:
        pass

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.logits], device=tensor.device)


def test_classifier_uses_only_densenet_checkpoint_and_labels(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    densenet_path = model_dir / "densenet121.pth"
    densenet_path.touch()
    manifest = {
        "class_names": ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"],
        "model_files": {
            "densenet": "models/densenet121.pth",
        },
    }
    (model_dir / prediction_module.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        prediction_module.models,
        "resnet50",
        lambda **_kwargs: pytest.fail("ResNet50 must not be loaded"),
    )
    monkeypatch.setattr(
        prediction_module.models,
        "densenet121",
        lambda **_kwargs: FakeNetwork("classifier", [0.0, 2.0, 0.0]),
    )
    loaded_paths = []

    def fake_load(path, **_kwargs):
        loaded_paths.append(path)
        return {}

    monkeypatch.setattr(prediction_module.torch, "load", fake_load)
    classifier = prediction_module.DenseNet121Classifier(model_dir, torch.device("cpu"))

    result = classifier.predict(Image.new("RGB", (8, 8)))

    expected = torch.tensor([0.0, 2.0, 0.0]).softmax(dim=0)
    assert loaded_paths == [densenet_path.resolve()]
    assert list(result) == ["Healthy", "Not Healthy", "Pneumonia"]
    assert list(result.values()) == pytest.approx(expected.tolist())