"""
One-off script: extract a small set of representative test engines and save
them as canned demo data for the Streamlit dashboard.

Run once: `python app/extract_demo_engines.py`
Outputs: app/example_data/demo_engines.json
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "app" / "example_data" / "demo_engines.json"

# Load test windows and labels
arrays = np.load(PROCESSED_DIR / "fd001_windows.npz")
X_test = arrays["X_test"]  # (100, 30, 14)
y_test = arrays["y_test"]  # (100,)
test_unit_ids = arrays["test_unit_ids"]  # (100,)

# Pick 6 engines: 2 critical, 2 watch, 2 healthy.
# Within each band, pick the median-RUL engine and one near-extreme so the
# dropdown has variety.


def pick_engines_in_band(low: int, high: int, n: int = 2) -> list[int]:
    """Return engine indices in [low, high) RUL band, evenly spaced by RUL."""
    band_mask = (y_test >= low) & (y_test < high)
    band_indices = np.where(band_mask)[0]
    if len(band_indices) == 0:
        return []
    # Sort by actual RUL, pick evenly-spaced
    sorted_by_rul = band_indices[np.argsort(y_test[band_indices])]
    if len(sorted_by_rul) <= n:
        return sorted_by_rul.tolist()
    # Pick n evenly spaced indices
    picks = np.linspace(0, len(sorted_by_rul) - 1, n, dtype=int)
    return sorted_by_rul[picks].tolist()


critical = pick_engines_in_band(0, 30)
watch = pick_engines_in_band(30, 80)
healthy = pick_engines_in_band(80, 126)

picked = critical + watch + healthy

regime_for = {}
for i in critical:
    regime_for[int(i)] = "Critical"
for i in watch:
    regime_for[int(i)] = "Watch"
for i in healthy:
    regime_for[int(i)] = "Healthy"

# Build the JSON structure
demo_engines = []
for idx in picked:
    demo_engines.append(
        {
            "test_index": int(idx),
            "engine_id": int(test_unit_ids[idx]),
            "actual_rul": float(y_test[idx]),
            "regime": regime_for[int(idx)],
            "window": X_test[idx].tolist(),  # (30, 14) → list of lists
        }
    )

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(demo_engines, f, indent=2)

print(f"Saved {len(demo_engines)} demo engines to {OUTPUT_PATH}")
print()
for e in demo_engines:
    print(
        f"  Engine #{e['engine_id']:3d}  test_idx={e['test_index']:3d}  "
        f"actual RUL={e['actual_rul']:6.1f}  regime={e['regime']}"
    )
