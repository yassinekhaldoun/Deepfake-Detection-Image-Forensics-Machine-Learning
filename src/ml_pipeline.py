"""Simple machine learning pipeline: feature loading, pair-wise split, models, evaluation.

Every feature extractor writes one CSV with the columns `image_id` and `label`,
followed by its own feature columns. A real image and its fake twin share the
same file name, so `image_id` is also the identifier of the PAIR.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One CSV per feature family, written by src/extract_all_features.py on the extended
# dataset. Add a line here when a new family is delivered (e.g. noise).
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FEATURE_FILES = {
    "baseline": FEATURES_DIR / "baseline.csv",
    "spectral": FEATURES_DIR / "spectral.csv",
    "rgb": FEATURES_DIR / "rgb.csv",
    "npr": FEATURES_DIR / "npr.csv",
}

KEYS = ["image_id", "label"]


# ---------------------------------------------------------------------------
# Step 1: load the features and split the data by pairs
# ---------------------------------------------------------------------------

def load_features(files=FEATURE_FILES):
    """Merge the feature tables of all families on (image_id, label).

    Returns:
        table: one row per image, with the columns image_id, label, pair_id, y
               (y = 1 for fake, 0 for real) and all feature columns.
        families: dict mapping each family name to the list of its feature columns.
    """
    table, families = None, {}
    for family, path in files.items():
        df = pd.read_csv(path)
        families[family] = [c for c in df.columns if c not in KEYS]
        if table is None:
            table = df
        else:
            table = table.merge(df, on=KEYS, how="inner", validate="one_to_one")

    all_columns = [c for columns in families.values() for c in columns]
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("Two families use the same feature name.")

    table["pair_id"] = table["image_id"]
    table["y"] = (table["label"] == "fake").astype(int)

    # A feature can be undefined (e.g. a completely white crop has no spectrum).
    # The whole pair is dropped, so the real/fake balance is kept.
    undefined = ~np.isfinite(table[all_columns]).all(axis=1)
    bad_pairs = table.loc[undefined, "pair_id"].unique()
    if len(bad_pairs):
        print(f"{len(bad_pairs)} pairs dropped: a feature is undefined for one of their images.")
        table = table[~table["pair_id"].isin(bad_pairs)].reset_index(drop=True)
    return table, families


def make_pair_split(table, test_size=0.3, seed=0):
    """Split the rows into train and test, keeping each real/fake pair together.

    The pairs (not the images) are shuffled with a fixed seed, so the split is
    reproducible and a real image never ends up in train while its fake twin is
    in test.
    """
    sizes = table.groupby("pair_id")["y"].agg(["count", "sum"])
    if not ((sizes["count"] == 2) & (sizes["sum"] == 1)).all():
        raise ValueError("Every pair must contain exactly one real and one fake image.")

    pairs = np.array(sorted(sizes.index))
    rng = np.random.default_rng(seed)
    pairs = rng.permutation(pairs)
    n_test = round(len(pairs) * test_size)
    test_pairs = set(pairs[:n_test])

    is_test = table["pair_id"].isin(test_pairs)
    return table[~is_test].reset_index(drop=True), table[is_test].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: baseline features and logistic regression
# ---------------------------------------------------------------------------

BASELINE_FEATURE_NAMES = ["base_mean", "base_std", "base_entropy", "base_grad_energy"]


def extract_baseline_features(image):
    """Four simple statistics of a 512x512 RGB image (uint8), used as a reference.

    They describe the image content, not forensic traces: if the forensic
    features do not beat them, they are not worth much.
    """
    if image.shape != (512, 512, 3):
        raise ValueError("Input image must have shape (512, 512, 3).")

    image = image.astype(np.float64)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    p = np.bincount(np.rint(gray).astype(int).ravel(), minlength=256) / gray.size
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    gy, gx = np.gradient(gray)
    grad_energy = float(np.mean(gx ** 2 + gy ** 2))

    return np.array([gray.mean(), gray.std(), entropy, grad_energy])


def train_logistic(train, columns, C=1.0):
    """Fit a standardized logistic regression on the given feature columns.

    The scaler is part of the model, so its mean and standard deviation are
    learned from the training rows only and then reused as they are on the test rows.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(train[columns], train["y"])
    return model


def predict_scores(model, data, columns):
    """Probability that each row is fake (between 0 and 1)."""
    return model.predict_proba(data[columns])[:, 1]


# ---------------------------------------------------------------------------
# Step 3: metrics and ablation
# ---------------------------------------------------------------------------

def evaluate(y_true, scores, threshold=0.5):
    """Accuracy, F1, AUC and confusion matrix counts for a set of scores.

    A row is predicted fake when its score is at least `threshold`.
    tn/fp/fn/tp = real kept as real / real flagged fake / fake missed / fake found.
    """
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted),
        "auc": roc_auc_score(y_true, scores),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# Which families each ablation row uses. A row is skipped if one of its families is missing.
ABLATION_GROUPS = {
    "baseline": ["baseline"],
    "spectral": ["spectral"],
    "rgb": ["rgb"],
    "npr": ["npr"],
    "spectral + rgb + npr": ["spectral", "rgb", "npr"],
}


# Progression: start from the basic statistics, then add one family at a time.
# Each row keeps everything of the previous row, so the gain of each family is visible.
PROGRESSION_GROUPS = {
    "baseline": ["baseline"],
    "+ spectral": ["baseline", "spectral"],
    "+ rgb": ["baseline", "spectral", "rgb"],
    "+ npr": ["baseline", "spectral", "rgb", "npr"],
}


def run_ablation(train, test, families, groups=ABLATION_GROUPS):
    """Train one logistic regression per group of families and evaluate it on the test rows.

    This shows what each family brings alone, and what the forensic families bring together.
    """
    rows = []
    for name, members in groups.items():
        if any(member not in families for member in members):
            continue
        columns = [c for member in members for c in families[member]]
        model = train_logistic(train, columns)
        scores = predict_scores(model, test, columns)
        rows.append({"features": name, "n_features": len(columns), **evaluate(test["y"], scores)})
    return pd.DataFrame(rows).set_index("features")


# ---------------------------------------------------------------------------
# Feature tables: compute the features of every image of the extended dataset
# ---------------------------------------------------------------------------
# Run once, from the project root:  python src/ml_pipeline.py [--limit N] [--workers W]
# It can be stopped and restarted: pairs already written are skipped.

REAL_DIR = PROJECT_ROOT / "data" / "extended" / "raise"
FAKE_DIR = PROJECT_ROOT / "data" / "extended" / "midjourney"

# Same order as extract_npr_features: channel, direction, then 4 statistics and 16 histogram bins.
NPR_FEATURE_NAMES = [
    f"npr_{channel}_{direction}_{stat}"
    for channel in "RGB"
    for direction in ("horizontal", "vertical", "diagonal")
    for stat in ["mean", "var", "skew", "kurt"] + [f"hist{i}" for i in range(16)]
]


@lru_cache(maxsize=None)
def _extractors():
    """Family name -> (extractor function, column names). Imported here to keep load_features light."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.spectral import SPECTRAL_FEATURE_NAMES, extract_spectral_features
    from rgb_method.rgb import FEATURE_NAMES as RGB_FEATURE_NAMES, extract_rgb_features
    from npr_method.npr import extract_npr_features
    return {
        "baseline": (extract_baseline_features, BASELINE_FEATURE_NAMES),
        "spectral": (extract_spectral_features, SPECTRAL_FEATURE_NAMES),
        "rgb": (extract_rgb_features, RGB_FEATURE_NAMES),
        "npr": (extract_npr_features, NPR_FEATURE_NAMES),
    }


def _process_pair(name):
    """Features of a real image and of its fake twin: {family: [row_real, row_fake]}."""
    from src.preprocessing import center_crop, load_image
    families = _extractors()
    rows = {family: [] for family in families}
    for label, folder in (("real", REAL_DIR), ("fake", FAKE_DIR)):
        image = center_crop(load_image(folder / name), 512)
        for family, (extractor, _) in families.items():
            rows[family].append([name, label, *extractor(image)])
    return rows


def build_feature_tables(limit=None, workers=4):
    """Compute the features of all pairs and append them to results/features/<family>.csv."""
    from multiprocessing import Pool

    families = _extractors()
    names = sorted(p.name for p in REAL_DIR.glob("*.png"))
    missing = [n for n in names if not (FAKE_DIR / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} real images have no fake twin, e.g. {missing[:3]}")

    marker = FEATURES_DIR / "baseline.csv"
    done = set(pd.read_csv(marker)["image_id"]) if marker.exists() else set()
    todo = [n for n in names if n not in done][:limit]
    print(f"{len(names)} pairs in total, {len(done)} already done, {len(todo)} to process", flush=True)

    def flush(buffer):
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        for family, (_, columns) in families.items():
            path = FEATURES_DIR / f"{family}.csv"
            df = pd.DataFrame(buffer[family], columns=KEYS + columns)
            df.to_csv(path, mode="a", header=not path.exists(), index=False)

    buffer = {family: [] for family in families}
    with Pool(workers) as pool:
        for i, rows in enumerate(pool.imap(_process_pair, todo), 1):
            for family in families:
                buffer[family].extend(rows[family])
            if i % 25 == 0 or i == len(todo):
                flush(buffer)
                buffer = {family: [] for family in families}
                print(f"{i}/{len(todo)} pairs written", flush=True)


# ---------------------------------------------------------------------------
# Detector: train once on all pairs, then classify any image file
# ---------------------------------------------------------------------------

DETECTOR_FILE = PROJECT_ROOT / "models" / "detector.joblib"
DETECTOR_FAMILIES = ("spectral", "rgb", "npr")   # forensic families only, no content statistics


def train_detector(table, families, family_names=DETECTOR_FAMILIES):
    """Fit the logistic regression on ALL rows of the table (no test set kept apart)."""
    columns = [c for name in family_names for c in families[name]]
    return {
        "model": train_logistic(table, columns),
        "columns": columns,
        "families": list(family_names),
        "n_train_pairs": int(table["pair_id"].nunique()),
    }


def save_detector(detector, path=DETECTOR_FILE):
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)


def load_detector(path=DETECTOR_FILE):
    import joblib
    return joblib.load(path)


def predict_image(path, detector):
    """Probability that an image file is AI-generated.

    The features are computed on the central 512x512 crop, exactly as for the training images.
    Raises ValueError if the image is smaller than 512x512 or if a feature is undefined.
    """
    extractors = _extractors()
    from src.preprocessing import center_crop, load_image

    image = center_crop(load_image(path), 512)
    values = {}
    for family in detector["families"]:
        extractor, names = extractors[family]
        values.update(zip(names, extractor(image)))
    row = pd.DataFrame([values])[detector["columns"]]
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("a feature is undefined for this image (e.g. a completely uniform crop)")
    return float(detector["model"].predict_proba(row)[0, 1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute the feature tables of the extended dataset.")
    parser.add_argument("--limit", type=int, default=None, help="process only N pairs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_feature_tables(args.limit, args.workers)
