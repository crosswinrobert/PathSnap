import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.data_io import load_trip
from src.viz import make_trip_map

TRIP_IDS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

OUTPUT_DIR = Path("experiments/api_evaluation/route_comparison_maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for trip_id in TRIP_IDS:
    print(f"Generating comparison map for {trip_id}...")

    request = Request(
        f"http://127.0.0.1:8000/predict/{trip_id}",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    predicted_arcs = result["predicted_route"]

    # Load reference route only for internal comparison.
    trip = load_trip(
        partition="student_test",
        trip_id=trip_id,
        load_route=True
    )

    output_path = OUTPUT_DIR / f"{trip_id}_comparison.html"

    make_trip_map(
        trip,
        out_html=str(output_path),
        predicted_arcs=predicted_arcs,
        show_network=False
    )

    print(f"Saved: {output_path}")

print("\nAll route comparison maps generated.")
