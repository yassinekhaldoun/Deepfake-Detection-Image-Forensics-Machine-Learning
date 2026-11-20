"""Tell whether an image looks AI-generated.

Usage:  python src/predict_image.py path/to/image.png

The detector is trained once on all the pairs of the extended dataset and saved in
models/detector.joblib; the next runs reuse it.
"""

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline import (DETECTOR_FILE, load_detector, load_features, predict_image,
                             save_detector, train_detector)


def get_detector():
    if DETECTOR_FILE.exists():
        return load_detector()
    print("Training the detector on all pairs (first run only)...")
    table, families = load_features()
    detector = train_detector(table, families)
    save_detector(detector)
    return detector


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    with Image.open(path) as image:
        image_format, (width, height) = image.format, image.size

    try:
        probability = predict_image(path, get_detector())
    except ValueError as error:
        raise SystemExit(f"Cannot analyse this image: {error}")

    verdict = "AI-GENERATED" if probability >= 0.5 else "REAL"
    print(f"{path.name}: {verdict} (probability of AI: {probability:.1%})")
    print(f"Format {image_format}, size {width}x{height}; analysed on the central 512x512 crop.")
    if image_format == "JPEG":
        print("Warning: JPEG compression erases the camera traces this detector relies on; "
              "a real photo can be wrongly flagged as AI.")
    print("Note: trained on native-resolution camera crops versus Midjourney images only; "
          "treat the answer as an indication.")


if __name__ == "__main__":
    main()
