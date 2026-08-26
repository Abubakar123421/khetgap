# External sugarcane imagery

The full external orthomosaic is intentionally excluded from version control.

Recommended research source:

- **Dataset:** Orthomosaic Dataset of RGB Aerial Images for Crop Rows Detection
- **Publisher:** LAPIX/UFSC (2019)
- **Page:** https://lapix.ufsc.br/crop-rows-sugar-cane/
- **Capture:** RGB Canon G9X on a fixed-wing UAV at 125-200 m
- **Published spatial resolution:** approximately 5 cm/pixel
- **Contents:** original orthomosaic and human-created crop-row ground truth

Before redistributing any original image or derived crop, confirm permission with
the publisher. Store locally downloaded files in this directory; `.gitignore`
keeps them out of commits. Record the downloaded filename and SHA-256 checksum
in experiment notes. Treat 0.05 m/pixel as approximate demo calibration, not a
survey-grade measurement.

Local verification download checksums:

- `sugarcane1.png`: `7F445C1043AEDACC78BA9361FC39F11B88C40A45F7C19ACACD82CCE2C94CC469`
- `sugarcane1-GT.png`: `BF60E4F80EEC839DE3082DA747243D2091A9B4D37AA1F10E6339028E4AE5E575`

`scripts/verify_demo.py` creates two untracked local crops from the orthomosaic
for clean and difficult real-image checks.

The repository's `data/synthetic/` fixtures are generated locally and are safe
to redistribute.
