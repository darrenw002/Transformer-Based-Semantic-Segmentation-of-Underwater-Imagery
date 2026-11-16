"""Report exact training/TEST image overlap and scores excluding those TEST images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

CLASSES = ["RO", "FV", "HD", "RI", "WR"]
MODELS = ["unet", "unet_resnet18", "segformer_b0", "unetformer_resnet18"]


def image_hashes(folder):
    groups = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path.stem)
    return groups


def category_means(table):
    rows = []
    for category in CLASSES:
        subset = table.loc[table["class"] == category]
        rows.append({"class": category, "scored_images": int(subset["boundary_f"].notna().sum()),
                     "boundary_f_pct": float(100 * subset["boundary_f"].mean()),
                     "iou_pct": float(100 * subset["iou"].mean())})
    return pd.DataFrame(rows)


def audit(root):
    train_folder = root / "cleaned_suim/train_val/images"
    if not train_folder.is_dir():
        train_folder = root / "train_val/images"
    train_groups = image_hashes(train_folder)
    test_groups = image_hashes(root / "TEST/images")
    partitions = {}
    split_path = root / "data/test/output/unet/split.csv"
    if split_path.is_file():
        with split_path.open(encoding="utf-8", newline="") as file:
            partitions = {row["stem"]: row["partition"] for row in csv.DictReader(file)}
    overlaps = []
    for digest in sorted(set(train_groups) & set(test_groups)):
        for test_stem in test_groups[digest]:
            for train_stem in train_groups[digest]:
                overlaps.append({"test_stem": test_stem, "train_val_stem": train_stem,
                                 "partition": partitions.get(train_stem, "unknown"),
                                 "image_sha256": digest})
    overlap_stems = {row["test_stem"] for row in overlaps}
    test_stems = {stem for group in test_groups.values() for stem in group}
    comparisons, categories = [], []
    for model in MODELS:
        folder = root / "data/test/output" / model
        metrics_path, summary_path = folder / "per_image_metrics.csv", folder / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        table = pd.read_csv(metrics_path, dtype={"stem": str, "class": str})
        if set(table["stem"]) != test_stems or table.duplicated(["stem", "class"]).any():
            raise ValueError(f"{model}: expected one complete evaluation of the original TEST set.")
        if len(table) != len(test_stems) * len(CLASSES):
            raise ValueError(f"{model}: expected all five category rows for every TEST image.")
        for protocol, selected in (
                ("original_test", table),
                ("exclude_exact_train_val_overlap", table.loc[~table.stem.isin(overlap_stems)])):
            means = category_means(selected)
            f_score, iou = means.boundary_f_pct.mean(), means.iou_pct.mean()
            if protocol == "original_test":
                if abs(f_score - summary["macro_boundary_f_pct"]) > 1e-8:
                    raise ValueError(f"{model}: per-image F-score aggregation disagrees with its summary.")
                if abs(iou - summary["macro_iou_pct"]) > 1e-8:
                    raise ValueError(f"{model}: per-image IoU aggregation disagrees with its summary.")
            comparisons.append({"model": model, "mode": summary["mode"], "protocol": protocol,
                                "test_images": selected.stem.nunique(),
                                "boundary_f_pct": f_score, "iou_pct": iou})
            for row in means.to_dict(orient="records"):
                categories.append({"model": model, "protocol": protocol, **row})
    return {
        "training_image_folder": str(train_folder),
        "train_val_images": sum(map(len, train_groups.values())),
        "test_images": len(test_stems),
        "overlapping_test_images": len(overlap_stems),
        "test_images_after_exclusion": len(test_stems - overlap_stems),
        "overlaps": overlaps,
        "scope": "Exact SHA-256 image-file matches only; no perceptual/near-duplicate claim.",
        "scoring": "Same saved predictions; unchanged upstream per-image/class averaging; no model retuning.",
    }, pd.DataFrame(comparisons), pd.DataFrame(categories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report, comparisons, categories = audit(root)
    output = root / "artifacts/split_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(report["overlaps"]).to_csv(output / "exact_test_overlaps.csv", index=False)
    comparisons.to_csv(output / "model_comparison.csv", index=False)
    categories.to_csv(output / "class_metrics.csv", index=False)
    print(f"Exact TEST overlaps: {report['overlapping_test_images']}/{report['test_images']}")
    print(f"TEST images after exclusion: {report['test_images_after_exclusion']}")
    print(comparisons.round(2).to_string(index=False))
    print("Saved audit:", output)


if __name__ == "__main__":
    main()
