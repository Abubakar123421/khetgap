"""Run the reproducible demo fixtures and save visual QA artifacts."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import CropConfig, analyze_field  # noqa: E402


INPUT = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "output" / "verification"
EXTERNAL = ROOT / "data" / "external"


def prepare_external_tiles() -> list[tuple[str, Path]]:
    source = EXTERNAL / "sugarcane1.png"
    if not source.exists():
        return []
    image = Image.open(source).convert("RGB")
    specifications = {
        "lapix_clean": (5600, 2200, 8200, 4400),
        "lapix_hard": (2500, 1000, 5000, 3200),
    }
    prepared = []
    for name, bounds in specifications.items():
        path = EXTERNAL / f"{name}.png"
        image.crop(bounds).save(path)
        prepared.append((name, path))
    return prepared


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    config = CropConfig(min_gap_px=35, occupancy_threshold=0.16)
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        [p for p in INPUT.iterdir() if p.suffix.lower() in supported_extensions]
    )
    for path in image_paths:
        name = path.stem
        image = np.asarray(Image.open(path).convert("RGB"))
        started = perf_counter()
        result = analyze_field(image, config)
        elapsed = perf_counter() - started
        Image.fromarray(result.overlay_image).save(OUTPUT / f"{name}_overlay.png")
        for debug_name in ("mask", "rotated_mask", "row_bands"):
            if debug_name in result.debug_images:
                Image.fromarray(result.debug_images[debug_name]).save(
                    OUTPUT / f"{name}_{debug_name}.png"
                )
        summaries[name] = {
            "status": result.status,
            "elapsed_seconds": round(elapsed, 4),
            "metrics": result.metrics,
            "warnings": result.warnings,
            "errors": result.errors,
            "gaps": [asdict(gap) for gap in result.gaps],
            "pattern_hints": result.pattern_hints,
        }
    real_config = CropConfig(
        exg_threshold=18,
        hsv_lower=(20, 20, 15),
        min_row_spacing_px=14,
        row_band_half_width_px=4,
        occupancy_threshold=0.12,
        min_gap_px=18,
    )
    for name, path in prepare_external_tiles():
        image = np.asarray(Image.open(path).convert("RGB"))
        started = perf_counter()
        result = analyze_field(image, real_config, meters_per_pixel=0.05)
        elapsed = perf_counter() - started
        Image.fromarray(result.overlay_image).save(OUTPUT / f"{name}_overlay.png")
        summaries[name] = {
            "status": result.status,
            "elapsed_seconds": round(elapsed, 4),
            "metrics": result.metrics,
            "warnings": [
                "The 0.05 m/pixel calibration is approximate and published by the dataset source.",
                *result.warnings,
            ],
            "errors": result.errors,
            "gap_count": len(result.gaps),
            "pattern_hints": result.pattern_hints,
        }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summaries, indent=2, default=json_default), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
