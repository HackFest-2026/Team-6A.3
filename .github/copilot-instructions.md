# Project Guidance

- Keep the ResNet50 + DenseNet121 ensemble as the only authority for Healthy, Not Healthy, and Pneumonia classification, in that order.
- Keep YOLO limited to localization: draw red boxes without labels and report the maximum detection confidence, or null when no boxes are found.
- Never add placeholder or fabricated predictions. The API must fail startup when required checkpoints are unavailable or incompatible.
- Preserve the 5 MB JPG/JPEG/PNG validation limit and runtime-only upload history.
- Keep frontend API calls typed and route them through `frontend/src/api.ts`.
- Update focused backend tests and this README when changing API behavior, checkpoint format, or setup requirements.
