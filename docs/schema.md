# ArchLens DiagramGraph Schema Contract (`v1.0`)

`DiagramGraph` (defined in `src/archlens/graph/models.py`) is the **single canonical contract** shared across all ArchLens pipeline stages:

- **Producers**: `detect`, `ocr`, `connectors`, and `graph` assemble and enrich `DiagramGraph`.
- **Consumers**: `review` evaluates architectural design rules directly against `DiagramGraph`, and `export` serializes `DiagramGraph` to JSON, Mermaid, and Graphviz DOT.

## Top-Level Model: `DiagramGraph`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `nodes` | `list[Node]` | `[]` | Detected architectural components. |
| `edges` | `list[Edge]` | `[]` | Detected connectors and arrows between nodes. |
| `metadata` | `GraphMetadata` | `GraphMetadata()` | Source image dimensions, stage timings, and provenance. |

### Referential & Uniqueness Invariants

`DiagramGraph` enforces the following invariants at validation time:

1. All `Node.id` values within `nodes` must be unique.
2. All `Edge.id` values within `edges` must be unique.
3. Every `Edge.source` and `Edge.target` must reference an existing `Node.id` in `nodes`.

---

## Component Model: `Node`

| Field | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `id` | `str` | required | Non-empty unique identifier (e.g., `"n1"`). |
| `type` | `"box" \| "database" \| "ellipse" \| "component"` | `"box"` | Canonical component shape category. |
| `bbox` | `BBox` | required | Pixel-space bounding box (`x >= 0`, `y >= 0`, `w > 0`, `h > 0`). |
| `label` | `str` | `""` | Associated OCR text label (empty string if unlabeled). |
| `confidence` | `float` | `1.0` | Confidence score in `[0.0, 1.0]`. |

### `BBox`

| Field | Type | Constraints |
| --- | --- | --- |
| `x` | `int` | `>= 0` (left edge in pixels) |
| `y` | `int` | `>= 0` (top edge in pixels) |
| `w` | `int` | `> 0` (width in pixels) |
| `h` | `int` | `> 0` (height in pixels) |

---

## Connector Model: `Edge`

| Field | Type | Default | Constraints / Notes |
| --- | --- | --- | --- |
| `id` | `str` | required | Non-empty unique identifier (e.g., `"e1"`). |
| `source` | `str` | required | Must match an existing `Node.id`. |
| `target` | `str` | required | Must match an existing `Node.id`. |
| `directed` | `bool` | `True` | `True` when an arrowhead indicates direction (`source -> target`). |
| `label` | `str \| None` | `None` | Optional OCR label attached to the connector. |
| `polyline` | `list[Point]` | `[]` | Ordered `{"x": float, "y": float}` points tracing the connector path. |
| `confidence` | `float` | `1.0` | Confidence score in `[0.0, 1.0]`. |

### `Point`

| Field | Type | Notes |
| --- | --- | --- |
| `x` | `float` | X coordinate in pixels. |
| `y` | `float` | Y coordinate in pixels. |

`Point` also accepts 2-element sequences `(x, y)` or `[x, y]` during validation.

---

## Metadata Model: `GraphMetadata`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image_size` | `ImageSize \| None` | `None` | Source image `{ "width": int (>0), "height": int (>0) }`. |
| `timings_ms` | `dict[str, float]` | `{}` | Per-stage execution timings in milliseconds. |
| `source_image` | `str \| None` | `None` | Input image path or filename. |
| `ocr_engine` | `str \| None` | `None` | Local OCR adapter identifier (e.g., `"tesseract"`, `"paddle"`). |
| `schema_version` | `str` | `"1.0"` | Contract version string. |

---

## Example JSON Payload

```json
{
  "nodes": [
    {
      "id": "client",
      "type": "ellipse",
      "bbox": { "x": 40, "y": 100, "w": 120, "h": 60 },
      "label": "Web Client",
      "confidence": 0.96
    },
    {
      "id": "api",
      "type": "box",
      "bbox": { "x": 260, "y": 95, "w": 150, "h": 70 },
      "label": "API Gateway",
      "confidence": 0.94
    },
    {
      "id": "db",
      "type": "database",
      "bbox": { "x": 520, "y": 90, "w": 110, "h": 80 },
      "label": "Users DB",
      "confidence": 0.91
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "client",
      "target": "api",
      "directed": true,
      "label": "HTTPS",
      "polyline": [
        { "x": 160.0, "y": 130.0 },
        { "x": 260.0, "y": 130.0 }
      ],
      "confidence": 0.92
    },
    {
      "id": "e2",
      "source": "api",
      "target": "db",
      "directed": true,
      "label": "SQL",
      "polyline": [
        { "x": 410.0, "y": 130.0 },
        { "x": 520.0, "y": 130.0 }
      ],
      "confidence": 0.89
    }
  ],
  "metadata": {
    "image_size": { "width": 800, "height": 400 },
    "timings_ms": { "preprocess": 8.2, "detect": 14.5, "ocr": 41.0 },
    "source_image": "sample_images/web_app.png",
    "ocr_engine": "tesseract",
    "schema_version": "1.0"
  }
}
```

