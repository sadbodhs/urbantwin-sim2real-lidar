"""Per-track constants of the UrbanTwin Sim2Real challenge (from the track pages and the starting kit)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TRACKS = {
    "lumpi": dict(
        classes=["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"],
        pc_range=[-40.0, -40.0, -5.0, 40.0, 40.0, 5.0],
        kit=REPO / "configs/starter_kit_mirror/lumpi",
    ),
    "v2x": dict(
        classes=["vehicle", "pedestrian", "truck"],
        pc_range=[-40.0, -40.0, -8.0, 40.0, 40.0, 2.0],
        kit=REPO / "configs/starter_kit_mirror/v2x_real",
    ),
}

# Normalisation endpoints, as in both official kits' config.json (LUMPI and V2X-Real).
NORM = {"CD": (2.0, 30.0), "MMD": (0.0, 0.5), "EMD": (0.0, 5.0), "FPD": (0.0, 50.0), "AP": (0.0, 0.7)}

DECLARATION = ("I confirm that I did not train on any frame listed in forbidden_frames.txt, and that all submitted "
               "detections were produced by a model trained only on synthetic data I generated.")


def test_frame_ids(track):
    return json.loads((TRACKS[track]["kit"] / "detection_test_frame_ids.json").read_text())
