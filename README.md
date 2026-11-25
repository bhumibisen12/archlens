# ArchLens

**ArchLens** is a local, privacy-first computer vision tool that reads photos or scans of hand-drawn or printed software-architecture diagrams, converts their components and labelled arrows into a structured directed graph (`DiagramGraph`), and runs deterministic, rule-based architectural design checks.

> **Status:** MVP in progress.

## Problem

Engineering teams regularly sketch system designs on whiteboards and paper during design reviews, then manually re-draw them into diagram tools and audit them by hand for architectural pitfalls (such as single points of failure, missing gateways, or direct client-to-database access). Sending unpublished architecture diagrams to hosted cloud vision APIs introduces privacy, IP, and compliance risks. ArchLens automates diagram digitization and preliminary design checks **entirely on your local machine**.

## Privacy Statement

- **100% local execution**: Image preprocessing, shape detection, OCR, connector tracing, graph export, and design checks all execute locally.
- **No cloud AI APIs, no VLMs, no runtime network calls**: The MVP uses classical OpenCV geometry and local OCR (`Tesseract` or offline `PaddleOCR` loaded from a pre-populated local model cache).
- **No telemetry**: No analytics, crash reporting, or usage tracking are sent externally.

## Installation

### 1. Install the Tesseract OCR system binary

ArchLens uses Tesseract via `pytesseract` as its default local OCR adapter. Install the system binary and ensure `tesseract` is on your `PATH`:

- **Windows**: Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) or run:
  ```powershell
  winget install UB-Mannheim.TesseractOCR
  ```
- **macOS**:
  ```bash
  brew install tesseract
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr
  ```

### 2. Install ArchLens (Python 3.10 or 3.11)

```bash
git clone https://github.com/bhumibisen12/archlens.git
cd archlens
python -m venv .venv
# Activate the virtual environment, then:
pip install -e .
```

Optional extras:

```bash
# Optional local PaddleOCR adapter (requires pre-downloaded local det/rec models):
pip install -e ".[paddle]"

# Development tools (ruff):
pip install -e ".[dev]"
```

## Usage

### CLI (`python -m archlens.cli`)

Parse a diagram image into JSON, Mermaid (`flowchart LR`), and Graphviz DOT outputs, with optional intermediate debug overlays:

```bash
python -m archlens.cli sample_images/test_0001.png --out result/ --ocr tesseract --debug
```

### Synthetic Diagram Generator

Generate labelled synthetic diagrams (`PNG` + ground-truth `DiagramGraph` `JSON`):

```bash
python scripts/generate_synthetic.py --n 20 --seed 42 --out data/synthetic --split train
```

### Evaluation & Ablations

Evaluate against synthetic or real labelled diagrams (see [docs/real_data.md](docs/real_data.md)):

```bash
python scripts/evaluate.py --data data/synthetic --out results/ --ocr tesseract
python scripts/evaluate.py --data data/synthetic --out results_no_persp/ --no-perspective
python scripts/evaluate.py --data data/synthetic --out results_no_shadow/ --no-shadow-removal
```

### Local Streamlit UI

```bash
streamlit run app/streamlit_app.py --browser.gatherUsageStats false
```

#### Screenshots

To capture UI screenshots for documentation:

1. Launch the Streamlit UI with `streamlit run app/streamlit_app.py --browser.gatherUsageStats false`.
2. Pick a sample from `sample_images/` or upload a diagram image, and toggle **Show Debug Views** in the sidebar as needed.
3. Capture screenshots of the **Visual Analysis**, **Extracted Architecture Graph**, and **Architectural Review Findings** sections and save them under `docs/screenshots/`.

## Repository Layout

```text
archlens/
  .github/workflows/ci.yml   # GitHub Actions CI (ruff + pytest on Python 3.10 & 3.11)
  app/streamlit_app.py       # Local Streamlit inspection & review UI
  configs/
    default.yaml             # Pipeline, detector, OCR, connector, and review thresholds
    roles.yaml               # Keyword-to-role inference mappings
  docs/
    architecture.md          # System architecture & Mermaid pipeline diagram
    dataset_card.md          # Dataset card template & consent/redaction guidelines
    limitations.md           # Documented classical CV limitations
    real_data.md             # Real-world evaluation dataset format
    schema.md                # Canonical DiagramGraph v1.0 contract
  sample_images/             # Tracked sample synthetic diagrams (PNG + JSON)
  scripts/
    evaluate.py              # Dataset evaluation & ablation CLI
    generate_synthetic.py    # Synthetic diagram dataset generator
  src/archlens/
    cli.py                   # python -m archlens.cli entrypoint
    config.py                # Pydantic configuration models & YAML loader
    logging_utils.py         # Local structured logging helpers
    pipeline.py              # End-to-end ArchLensPipeline orchestrator
    preprocess/              # Resize, perspective warp, shadow removal, denoise, threshold
    detect/                  # ComponentDetector ABC & ClassicalDetector
    ocr/                     # OCREngine ABC, Tesseract, offline PaddleOCR, text assigner
    connectors/              # ConnectorExtractor ABC, skimage skeleton tracer, arrowhead probe
    graph/                   # DiagramGraph Pydantic schema & JSON I/O
    review/                  # Role inference & deterministic design rule engine
    export/                  # JSON, Mermaid flowchart LR, and Graphviz DOT exporters
    evaluation/              # Hungarian IoU >= 0.5 matching, edit similarity, CSV/MD reports
  tests/                     # Pytest suite covering every module
```

## Results

Run `python scripts/evaluate.py --data <dataset_dir> --out results/` to generate `results/metrics.csv`, `results/report.md`, and `results/worst_overlays/`, then record measured runs below:

| Split / Run | OCR Engine | Ablation | Node P / R / F1 | Directed Edge P / R / F1 | Undirected Edge F1 | Label Sim | Exact Match Rate | Mean Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _TBD_ | _TBD_ | _None_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Known Limitations

- Dense crossing connectors, dashed/dotted lines, tightly curved/looping arrows, thin open arrowheads, and edge labels drawn directly over strokes can split or mis-link connectors. See [docs/limitations.md](docs/limitations.md) for details.

## Roadmap

- **Learned detector adapter**: Implement a local ONNX/PyTorch detector behind `archlens.detect.ComponentDetector` to improve recall and shape classification on messy sketches and dense crossings.
- **Handwriting OCR**: Evaluate and tune local handwriting-capable OCR models (behind `archlens.ocr.OCREngine`) for freehand whiteboard lettering.
- **VLM baseline comparison**: Benchmark local classical + learned pipelines against offline VLM baselines using `scripts/evaluate.py` on shared evaluation sets without adding runtime cloud dependencies.

## Contributing & License

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and [docs/dataset_card.md](docs/dataset_card.md) for dataset consent and redaction standards.
- Licensed under the [MIT License](LICENSE).

## Results

Run `python scripts/evaluate.py --data <dataset_dir> --out results/` to generate `results/metrics.csv`, `results/report.md`, and `results/worst_overlays/`, then record measured runs below:

| Split / Run | OCR Engine | Ablation | Node P / R / F1 | Directed Edge P / R / F1 | Undirected Edge F1 | Label Sim | Exact Match Rate | Mean Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _TBD_ | _TBD_ | _None_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Known Limitations

- Dense crossing connectors, dashed/dotted lines, tightly curved/looping arrows, thin open arrowheads, and edge labels drawn directly over strokes can split or mis-link connectors. See [docs/limitations.md](docs/limitations.md) for details.

## Roadmap

- **Learned detector adapter**: Implement a local ONNX/PyTorch detector behind `archlens.detect.ComponentDetector` to improve recall and shape classification on messy sketches and dense crossings.
- **Handwriting OCR**: Evaluate and tune local handwriting-capable OCR models (behind `archlens.ocr.OCREngine`) for freehand whiteboard lettering.
- **VLM baseline comparison**: Benchmark local classical + learned pipelines against offline VLM baselines using `scripts/evaluate.py` on shared evaluation sets without adding runtime cloud dependencies.

## Contributing & License

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and [docs/dataset_card.md](docs/dataset_card.md) for dataset consent and redaction standards.
- Licensed under the [MIT License](LICENSE).

## Verification

```bash
ruff check .
pytest -v
```
