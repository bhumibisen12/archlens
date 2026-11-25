# Real-World Evaluation Dataset Format

`scripts/evaluate.py` and `src/archlens/evaluation/` evaluate both synthetic and real-world diagram photos using the same `DiagramGraph` contract (`docs/schema.md`).

## Directory Layout

Place each image alongside a matching `.json` ground-truth file sharing the same filename stem:

```text
my_real_dataset/
  whiteboard_01.jpg
  whiteboard_01.json
  notebook_02.png
  notebook_02.json
```

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`.

## Ground-Truth JSON Requirements

Each `.json` file must validate against `archlens.graph.DiagramGraph`:

- **`nodes`**: List of `Node` entries (`id`, `type` in `box | database | ellipse | component`, `bbox: {x, y, w, h}`, `label`).
  - Bounding boxes must be in **original image pixel coordinates** (`x >= 0, y >= 0, w > 0, h > 0`).
  - If perspective correction is enabled on angled photos-on-desk, note that `evaluate_dataset` matches predicted boxes against ground-truth coordinates directly; for uncropped angled photos, keep `--no-perspective` or annotate in the rectified frame.
- **`edges`**: List of `Edge` entries (`id`, `source`, `target`, `directed`, optional `label`, optional `polyline`).
  - `source` and `target` must reference valid `Node.id` entries in `nodes`.
- **`metadata`**: Optional `image_size` (`width`, `height`) and `source_image`.

## Running Evaluation & Ablations

```bash
# Baseline run
python scripts/evaluate.py --data my_real_dataset --out results --ocr tesseract

# Ablations
python scripts/evaluate.py --data my_real_dataset --out results_no_persp --no-perspective
python scripts/evaluate.py --data my_real_dataset --out results_no_shadow --no-shadow-removal
```

Outputs written to `--out`:
- `metrics.csv` — per-image node/edge P/R/F1, label similarity, exact-match flag, and stage latencies.
- `report.md` — dataset summary table, mean per-stage latency table, and worst-10 failure table.
- `worst_overlays/` — visual overlays (predicted nodes/arrows + thin green ground-truth boxes) for the worst-10 images.

