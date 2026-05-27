# Daily Summary - 2026-05-27

## Code Changes

- Added `third_party/ScalSelect/mits_tools/create_external_traffic_qa_template.py`.
  - Generates an Excel template for external traffic QA annotation.
  - Provides dropdown options for scene, task, target, and notes fields.
  - Includes example annotation rows and a guide sheet for field conventions.
  - Supports configurable output path and blank row count.

- Updated `third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py`.
  - Added validation for absolute local model paths before loading Qwen2.5-VL.
  - Reports clearer errors when a merged model directory is missing, is not a directory, or lacks `config.json`.
  - Keeps the existing HuggingFace shard validation flow intact.

## Generated Asset

- Added `external_traffic_qa_template.xlsx`.
  - Ready-to-use annotation workbook for external traffic QA data collection.
  - Contains an `annotations` sheet with example rows and blank rows for labeling.
  - Contains a `guide` sheet describing field formats, target naming, and note conventions.

## Upload Notes

- Local IDE files, logs, SwanLab logs, and Python cache files remain ignored by `.gitignore`.
- This upload focuses on today's annotation-template tooling and model-path validation changes.
