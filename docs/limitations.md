# Known Limitations (ArchLens MVP)

This document records the known geometric and computer-vision limitations of the classical MVP pipeline in `src/archlens/connectors/` and `src/archlens/detect/`.

## 1. Crossing Lines & Complex Junctions
- **4-way crossings (`+` / `X`)**: `ClassicalConnectorExtractor` resolves 4-way skeleton junctions by pairing opposite branches whose incoming/outgoing tangents align within `max_crossing_angle_deg`. If two arrows cross at a shallow angle, merge briefly before separating, or form a cluster of 3+ overlapping connectors at one junction, branches may be mispaired or split into unlinked segments.
- **Dense arrow loops**: Two crossing elbow arrows that enclose an empty white-paper pocket can confuse contour detectors when text happens to overlap the pocket.

## 2. Dashed & Dotted Lines
- Skeletonization operates on connected binary foreground ink. Dashed or dotted connectors break into short disconnected segments shorter than `min_polyline_length_px`, so they are not traced as a single continuous edge between nodes. Supporting dashed connectors would require collinear segment linking or morphological bridging along dash directions.

## 3. Very Curved or Hand-Drawn Looping Arrows
- `simplify_polyline` uses `cv2.approxPolyDP` with a fixed `simplify_epsilon_px`, and junction continuation (`_merge_crossing_chains`) assumes locally straight tangents near crossings. Tight S-curves, self-crossing loops, or steeply curved arcs may be over-simplified, lose accurate endpoint tangents for arrowhead probing, or fail tangent-continuation pairing at junctions.

## 4. Open / Hollow Arrowheads & Tiny Arrowheads
- Direction detection (`_detect_direction`) relies on ink density and local stroke thickness (distance transform) inside an oriented corridor near each endpoint. Filled triangular arrowheads produce a strong signal, whereas thin open `V`-shaped arrowheads or very small arrowheads may fall below `arrowhead_dominance_ratio` and be emitted as `directed=False`.

## 5. Edge Labels Overlapping Arrow Lines
- Masking OCR `TextBox` regions before skeletonization prevents edge-label text from generating spurious skeleton branches. If an edge label directly intersects the connector stroke itself, masking the text box can introduce a gap larger than 1 pixel and break the skeleton trace.
