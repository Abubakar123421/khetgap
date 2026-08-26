# KhetGap core engine

KhetGap converts one RGB aerial field image into detected sugarcane rows,
planting-gap polygons, an original-image overlay, and calibrated measurements.
This repository contains Muhammad Abubakar's UI-independent computer-vision and
backend deliverable. A Streamlit interface can treat `analyze_field()` as a
black box.

## Setup and verification

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\generate_synthetic_data.py
.venv\Scripts\python scripts\verify_demo.py
```

## Public API

```python
from PIL import Image
import numpy as np

from src import CropConfig, analyze_field

image = np.asarray(Image.open("data/synthetic/sugarcane_clean.png").convert("RGB"))
result = analyze_field(
    image=image,
    config=CropConfig(),
    meters_per_pixel=None,
    overrides={"row_angle_deg": None, "roi_xyxy": None},
)

if result.ok:
    print(result.metrics)
    Image.fromarray(result.overlay_image).save("output/overlay.png")
else:
    print(result.status, result.errors)
```

Inputs must be RGB `uint8` arrays shaped `H x W x 3`. The stable result contains
`status`, `overlay_image`, `metrics`, `gaps`, `pattern_hints`, `debug_images`,
`warnings`, and `errors`. Expected failures use `invalid_input`,
`no_vegetation`, or `no_rows`; they do not depend on UI code.

## Configuration and measurement

The reproducible sugarcane defaults are in `configs/crops.yaml`. Advanced values
remain tunable through `CropConfig`, including segmentation, morphology, row
spacing, occupancy, and gap thresholds. A manual row angle takes precedence over
automatic Hough/projection estimation.

Pixel measurements are always available for successful analyses. Metre lengths
and planting-material estimates are `None` unless a positive
`meters_per_pixel` value is supplied. `planting_units_per_meter` is a user-defined
demo rate, not a universal agronomic recommendation, and the safety factor
defaults to `1.0`.

## Image data and limitations

Synthetic fixtures are generated locally and include clean, noisy/rotated, and
negative examples. `data/external/README.md` records the provisional LAPIX/UFSC
sugarcane orthomosaic source without redistributing it.

The MVP assumes one dominant, approximately straight row direction in one ROI.
It does not diagnose agronomic causes, support curved/multiple-direction rows,
train a neural network, or produce georeferenced coordinates. Pattern messages
are deliberately phrased as possible explanations.
