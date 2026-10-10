from pathlib import Path

import pytest

from app.bounding import SuspiciousRegionDetector


def test_detector_loads_available_controlled_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "controlled_v1_best.pt"
    checkpoint.touch()
    loaded_paths: list[str] = []

    class FakeYOLO:
        def __init__(self, checkpoint_path: str) -> None:
            loaded_paths.append(checkpoint_path)

    monkeypatch.setattr("app.bounding.YOLO", FakeYOLO)

    SuspiciousRegionDetector(tmp_path, "cpu")

    assert loaded_paths == [str(checkpoint)]


def test_detector_fails_when_checkpoint_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="controlled_v1_best.pt"):
        SuspiciousRegionDetector(tmp_path, "cpu")