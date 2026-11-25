# ArchLens Architecture

ArchLens converts photos and scans of software architecture diagrams into validated `DiagramGraph` objects and deterministic design-review findings using an entirely local, modular computer vision pipeline.

## Pipeline Overview

```mermaid
flowchart LR
    IMG["Diagram Image (PNG/JPG)"] --> PP["preprocess: Resize, Perspective, Shadow Removal, Denoise, Adaptive Threshold"]
    PP -->|"color, gray, binary"| DET["detect: ComponentDetector (ClassicalDetector)"]
    PP -->|"gray"| OCR["ocr: OCREngine (TesseractEngine / offline PaddleOCREngine)"]
    PP -->|"binary"| CONN["connectors: ConnectorExtractor (Skeleton + Pixel Graph + Arrowhead Probe)"]
    DET -->|"Nodes (bbox, type, conf)"| CONN
    OCR -->|"TextBoxes (text, bbox, conf)"| CONN
    DET -->|"Nodes"| ASSIGN["ocr.assigner: Text-to-Node & Edge-Candidate Assignment"]
    OCR -->|"TextBoxes"| ASSIGN
    CONN -->|"Edges (polyline, directed, conf)"| ASSIGN
    ASSIGN --> GRAPH["graph: DiagramGraph (v1.0 Schema)"]
    GRAPH --> REVIEW["review: ReviewEngine + configs/roles.yaml"]
    GRAPH --> EXPORT["export: JSON / Mermaid / Graphviz DOT"]
    GRAPH --> EVAL["evaluation: Hungarian IoU >= 0.5 & Edit Similarity"]
```

## Module Responsibilities & Swappable Interfaces

| Module | Responsibility | Extension / Swap Point |
| --- | --- | --- |
| `archlens.preprocess` | Resizing, optional paper-quad perspective warp, illumination division, denoising, adaptive thresholding | `PreprocessConfig` in `configs/default.yaml` |
| `archlens.detect` | Component detection (`box`, `database`, `ellipse`, `component`) and nested duplicate filtering | `ComponentDetector` ABC (`detect(preprocessed) -> list[Node]`) |
| `archlens.ocr` | Local text recognition, configurable text normalization, and node/edge text assignment | `OCREngine` ABC (`read(image) -> list[TextBox]`) |
| `archlens.connectors` | Node/text masking, `skimage` skeletonization, pixel-graph tracing, snapping, arrowhead direction | `ConnectorExtractor` ABC (`extract(preprocessed, nodes, text_boxes) -> list[Edge]`) |
| `archlens.graph` | Canonical Pydantic contract (`Node`, `Edge`, `DiagramGraph`) and JSON I/O | Schema documented in [schema.md](schema.md) |
| `archlens.review` | Keyword role inference (`configs/roles.yaml`) and deterministic design rule evaluation | `ReviewRule` ABC (`evaluate(context) -> list[Finding]`) |
| `archlens.export` | Lossless JSON, Mermaid `flowchart LR`, and Graphviz DOT serialization | Pure functions over `DiagramGraph` |
| `archlens.evaluation` | Hungarian node matching (`IoU >= 0.5`), normalized label similarity, edge F1, latency & overlays | `evaluate_dataset(...)` and `scripts/evaluate.py` |

## Privacy & Offline Guarantees

- **No cloud AI APIs or VLMs** in the MVP runtime path.
- **No runtime network calls or telemetry**: Tesseract executes via local CLI, `PaddleOCREngine` requires pre-populated local `det/` and `rec/` directories (`DISABLE_MODEL_SOURCE_CHECK=True`, `PADDLEOCR_OFFLINE=1`), and the Streamlit UI runs locally.

