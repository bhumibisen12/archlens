# ArchLens Dataset Card Template

Use this template to document any synthetic or real-world diagram evaluation dataset collected for ArchLens. See [real_data.md](real_data.md) for the required directory layout and `DiagramGraph` JSON schema.

## 1. Dataset Details

- **Dataset Name**: `<e.g., ArchLens Whiteboard Dev Set v1>`
- **Version / Date**: `<YYYY-MM-DD>`
- **Maintainer**: `<Name / GitHub handle>`
- **Source Type**: `<Hand-drawn whiteboard | Paper notebook | Printed diagram | Synthetic>`
- **Sample Count**: `<N images + N ground-truth JSON files>`
- **License**: `<e.g., CC-BY-4.0 / Internal Only>`

## 2. Intended Use

- Intended for local evaluation, ablation testing (`scripts/evaluate.py`), and failure analysis of ArchLens component detection, OCR, and connector extraction.
- Not intended for training or redistributing proprietary system diagrams without explicit authorization.

## 3. Consent, Privacy & Redaction Guidelines

Before capturing, storing, or committing any real-world diagram photo or scan:

1. **Explicit Consent**: Obtain documented consent from the diagram's author(s) and, for workplace/client whiteboards, written approval from the system owner before photographing or storing the diagram.
2. **No Sensitive Information**: Inspect every image and ground-truth label to ensure no proprietary business secrets, customer PII, internal hostnames/IPs, API keys, passwords, or confidential project names appear in the image or `.json` file. Redact or crop out sensitive regions prior to annotation.
3. **Metadata Scrubbing**: Strip EXIF metadata (camera GPS coordinates, device serial numbers, timestamps) from photos before archiving or sharing.
4. **Withdrawal / Deletion**: Record a contact point so contributors or diagram authors can request immediate removal of their samples from the dataset.

## 4. Annotation Protocol

- Annotate all visible nodes (`box`, `database`, `ellipse`, `component`) with pixel-space `BBox` coordinates (`{x, y, w, h}`) and exact text labels matching [schema.md](schema.md).
- Annotate all connectors (`source`, `target`, `directed`, optional `label`, and optional `polyline`).
- Document inter-annotator review or edge-case decisions (e.g., overlapping arrows, ambiguous arrowheads).

## 5. Known Biases & Limitations

- Lighting conditions, camera angles, marker colors, and handwriting legibility represented in this dataset: `<describe>`
- Diagram styles underrepresented or excluded (e.g., swimlanes, sequence diagrams, dashed arrows): `<describe>`

