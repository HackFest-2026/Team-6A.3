from pathlib import Path
import argparse
import csv
import sys

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "models" / "controlled_v1_best.pt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main():
    parser = argparse.ArgumentParser(
        description="AS4 Controlled v1 YOLO11s lung-opacity detector (research prototype)"
    )
    parser.add_argument("input", help="Image file or folder of images")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help=f"Checkpoint path (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25; not the final-test metric threshold)")
    parser.add_argument("--iou", type=float, default=0.70,
                        help="NMS IoU threshold (default: 0.70)")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--output", help="Optional CSV output path")
    parser.add_argument("--save-dir", default="prediction_results",
                        help="Directory for annotated images")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if not weights_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {weights_path}\n"
            "Copy Controlled v1 best.pt into models/controlled_v1_best.pt "
            "or pass --weights with the correct path."
        )
    if not 0 <= args.conf <= 1:
        raise SystemExit("--conf must be between 0 and 1.")
    if not 0 < args.iou <= 1:
        raise SystemExit("--iou must be greater than 0 and at most 1.")
    if args.imgsz < 32:
        raise SystemExit("--imgsz must be at least 32.")

    files = collect_images(input_path)
    if not files:
        raise SystemExit("No supported images found (PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP).")

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable. Use --device cpu or auto.")
        device = "0"
    elif args.device == "cpu":
        device = "cpu"
    else:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    print("AS4 Controlled v1 YOLO inference")
    print(f"Device: {device}; images: {len(files)}")
    print(f"Image size: {args.imgsz}; confidence: {args.conf}; NMS IoU: {args.iou}")
    print("Research prototype only — not for medical diagnosis or treatment.")

    model = YOLO(str(weights_path))
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, image_path in enumerate(files, start=1):
        try:
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                save=False,
                verbose=False,
            )[0]

            # Save an annotated copy without modifying the source image.
            annotated = result.plot()
            output_image = save_dir / f"{index:05d}_{image_path.stem}.jpg"
            import cv2
            if not cv2.imwrite(str(output_image), annotated):
                raise RuntimeError(f"Could not write annotated image: {output_image}")

            boxes = result.boxes
            n = 0 if boxes is None else len(boxes)
            print(f"[{index}/{len(files)}] {image_path.name}: {n} detection(s)")

            if n == 0:
                rows.append({
                    "image": str(image_path),
                    "class_id": "",
                    "class_name": "",
                    "confidence": "",
                    "x1": "", "y1": "", "x2": "", "y2": "",
                    "annotated_image": str(output_image),
                })
            else:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confs = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for coords, score, class_id in zip(xyxy, confs, classes):
                    class_id = int(class_id)
                    names = result.names
                    class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    rows.append({
                        "image": str(image_path),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(score),
                        "x1": float(coords[0]), "y1": float(coords[1]),
                        "x2": float(coords[2]), "y2": float(coords[3]),
                        "annotated_image": str(output_image),
                    })
        except Exception as exc:
            print(f"ERROR: {image_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.output:
        csv_path = Path(args.output).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "annotated_image"
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {csv_path}")

    print(f"Annotated images saved under: {save_dir}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
