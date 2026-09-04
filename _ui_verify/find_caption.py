from pathlib import Path
root = Path(r".venv/Lib/site-packages/streamlit/static/static/js")
needles = ["stImageCaption", "stCaption", "stMetricLabel", "stFigCaption"]
for p in root.glob("*.js"):
    t = p.read_text(encoding="utf-8", errors="ignore")
    hits = [n for n in needles if n in t]
    if hits:
        print(p.name, hits)
