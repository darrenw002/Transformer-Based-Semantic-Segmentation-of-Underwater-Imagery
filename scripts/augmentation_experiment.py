"""Run matched no-augmentation trials using the existing SUIM model notebooks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "unet_resnet18": "05_unet_resnet18_control.ipynb",
    "segformer_b0": "03_segformer_baseline.ipynb",
    "unetformer_resnet18": "04_unetformer_baseline.ipynb",
}
CLASSES = ["RO", "FV", "HD", "RI", "WR"]
PRIMARY = "exclude_exact_train_val_overlap"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_sha256(path):
    # Model notebooks sort these columns and save LF line endings before hashing.
    table = pd.read_csv(path, dtype=str)[["stem", "image_sha256", "partition"]].sort_values("stem")
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def report_dir(root):
    return Path(root) / "artifacts/augmentation_experiment"


def output_dir(root, model, *, augmented):
    return Path(root) / "data/test/output" / (model if augmented else model + "_noaug")


def validate_pair(baseline, trial):
    """Reject a comparison if training/evaluation settings differ beyond augmentation."""
    shared = (
        "model", "training_directory", "test_directory", "train_pairs", "validation_pairs",
        "split_sha256", "classes", "image_size_width_height", "threshold", "seed",
        "epochs_requested", "image_mean", "image_std", "metric_protocol", "device",
        "inference_precision", "torch_version", "numpy_version",
    )
    for key in shared:
        if baseline.get(key) != trial.get(key):
            raise ValueError(f"Unmatched {key}: {baseline.get(key)!r} != {trial.get(key)!r}")
    for key in ("architecture", "auxiliary_loss_weight", "smp_version", "timm_version",
                "torchvision_version", "transformers_version"):
        if baseline.get(key) != trial.get(key):
            raise ValueError(f"Unmatched {key}")
    before, after = baseline["checkpoint_metadata"], trial["checkpoint_metadata"]
    changing = {"epoch", "val_loss", "use_augmentation"}
    if set(before) != set(after):
        raise ValueError("Checkpoint metadata fields differ")
    for key in set(before) - changing:
        if before[key] != after[key]:
            raise ValueError(f"Unmatched checkpoint {key}")
    if before["use_augmentation"] is not True or after["use_augmentation"] is not False:
        raise ValueError("Expected an augmented baseline and an unaugmented trial")
    for summary in (baseline, trial):
        if (summary["mode"] != "train" or summary["epochs_requested"] != 50
                or summary["test_images_evaluated"] != 110 or summary["partial_test"]):
            raise ValueError("Comparison requires completed 50-epoch training and full TEST evaluation")


def baseline_files(root):
    """Protect existing notebooks, checkpoints and evaluation reports while trials run."""
    paths = []
    for model, filename in MODELS.items():
        paths.append(Path(root) / "notebooks" / filename)
        paths.extend(p for p in output_dir(root, model, augmented=True).rglob("*") if p.is_file())
        paths.extend(p for p in (Path(root) / "ckpt" / model).rglob("*") if p.is_file())
    return sorted(set(paths))


def protected_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in baseline_files(root)}


def check_protected(root, expected):
    for relative, digest in expected.items():
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"A protected baseline file changed: {relative}")


def preflight(root=ROOT):
    root = Path(root).resolve()
    summaries = {}
    for model in MODELS:
        folder = output_dir(root, model, augmented=True)
        summary = read_json(folder / "summary.json")
        if (summary["model"] != model or summary["mode"] != "train"
                or summary["epochs_requested"] != 50 or summary["train_pairs"] != 1220
                or summary["validation_pairs"] != 305 or summary["test_images_evaluated"] != 110
                or summary["checkpoint_metadata"]["use_augmentation"] is not True):
            raise ValueError(f"{model}: completed augmented baseline required")
        weights = root / "ckpt" / model / "best.pt"
        if sha256(weights) != summary["weights_sha256"]:
            raise ValueError(f"{model}: checkpoint differs from the saved evaluation")
        history = pd.read_csv(weights.parent / "training_history.csv")
        if history.epoch.tolist() != list(range(1, 51)):
            raise ValueError(f"{model}: expected 50 completed baseline epochs")
        best = history.loc[history.val_loss.idxmin()]
        if (int(best.epoch) != summary["checkpoint_metadata"]["epoch"]
                or abs(best.val_loss - summary["checkpoint_metadata"]["val_loss"]) > 1e-10):
            raise ValueError(f"{model}: best checkpoint and history disagree")
        if split_sha256(root / "data/test/output/unet/split.csv") != summary["split_sha256"]:
            raise ValueError(f"{model}: the shared split changed")
        summaries[model] = summary
    return summaries


def build_trial_notebook(root, model, baseline, *, smoke_folder=None):
    root = Path(root).resolve()
    notebook = nbformat.read(root / "notebooks" / MODELS[model], as_version=4)
    source = {c.metadata.stage: c.source for c in notebook.cells if c.cell_type == "code"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop("suim_validation", None)
    destination = output_dir(root, model, augmented=False)
    checkpoint = root / "ckpt" / (model + "_noaug")
    if smoke_folder is not None:
        destination = Path(smoke_folder) / model / "output"
        checkpoint = Path(smoke_folder) / model / "ckpt"
    meta = baseline["checkpoint_metadata"]
    expected = {
        "EPOCHS": 50, "SEED": baseline["seed"], "CLASSES": baseline["classes"],
        "WIDTH": baseline["image_size_width_height"][0], "HEIGHT": baseline["image_size_width_height"][1],
        "BATCH_SIZE": 2, "LEARNING_RATE": meta["learning_rate"], "WEIGHT_DECAY": meta["weight_decay"],
        "IMAGE_MEAN": baseline["image_mean"], "IMAGE_STD": baseline["image_std"],
        "PREDICTION_THRESHOLD": baseline["threshold"], "AUGMENTATION": meta["augmentation"],
        "AMP_ENABLED": meta["mixed_precision_training"], "MAX_TEST_IMAGES": None,
    }
    overrides = (
        "\n# Augmentation experiment: reuse the model pipeline, change only this treatment and paths.\n"
        f"_expected_baseline_settings = {expected!r}\n"
        "for _name, _value in _expected_baseline_settings.items():\n"
        "    if globals()[_name] != _value:\n"
        "        raise ValueError(f'Baseline setting changed: {_name}')\n"
        "assert DEVICE.type == 'cuda', 'Use the verified PyTorch GPU environment'\n"
        "MODE = 'train'\nUSE_AUGMENTATION = False\n"
        f"OUTPUT_DIR = Path({str(destination)!r})\nCKPT_DIR = Path({str(checkpoint)!r})\n"
        "CUSTOM_WEIGHTS = CKPT_DIR / 'best.pt'\n"
        "print('EXPERIMENT: augmentation OFF; fresh ImageNet initialization; output:', OUTPUT_DIR)\n"
    )
    code = {c.metadata.stage: c for c in notebook.cells if c.cell_type == "code"}
    code["config"].source += overrides
    # Check the shared data and actual loader before any model training.
    code["inventory"].source += (
        f"\nassert SPLIT_SHA256 == {baseline['split_sha256']!r}\n"
        "assert len(train_stems) == 1220 and len(validation_stems) == 305\n")
    if smoke_folder is not None:
        code["config"].source += "\nEPOCHS = 1\nMAX_TEST_IMAGES = 2\n"
        code["inventory"].source += "\ntrain_stems = train_stems[:2]\nvalidation_stems = validation_stems[:2]\n"
    code["loader"].source += (
        "\nassert training_data.augment is False\nassert validation_data.augment is False\n"
        "print('Verified: training and validation augmentation are disabled')\n")
    # The source notebooks' introductory text describes their original augmented run.
    notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
        f"# {model}: no-augmentation experiment\n\n"
        "Executed copy of the existing model notebook. Configuration overrides below disable "
        "training augmentation and use separate output/checkpoint folders. Model, training and "
        "evaluation cells remain unchanged. Original notebook documentation is retained below; "
        "references to augmentation describe the source pipeline, while this trial disables it."))
    notebook.metadata["augmentation_experiment"] = {
        "model": model, "use_augmentation": False,
        "source_notebook": MODELS[model],
        "source_code_sha256": {stage: hashlib.sha256(text.encode()).hexdigest()
                               for stage, text in source.items()},
        "baseline_weights_sha256": baseline["weights_sha256"], "smoke": smoke_folder is not None,
        "completed": False,
    }
    nbformat.validate(notebook)
    return notebook, destination, checkpoint


def run_trial(root, model, baseline, *, smoke_folder=None):
    from nbclient import NotebookClient

    root = Path(root).resolve()
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    # All initial encoder weights are already cached; keep training independent of network availability.
    os.environ["HF_HUB_OFFLINE"] = "1"
    notebook, output, checkpoint = build_trial_notebook(root, model, baseline, smoke_folder=smoke_folder)
    saved = ((Path(smoke_folder) / f"{model}.ipynb") if smoke_folder is not None
             else report_dir(root) / "runs" / f"{model}_noaug.ipynb")
    saved.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, saved)
    started = time.time()

    class ProgressClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get("msg_type") == "stream" and msg["content"].get("name") == "stdout":
                for line in msg["content"].get("text", "").splitlines():
                    if line.startswith(("Epoch ", "Predicted ", "Macro ", "Verified:")):
                        print(f"[{model}] {line}", flush=True)
            return super().process_message(msg, cell, cell_index)

    def progress(**kwargs):
        cell = kwargs["cell"]
        if cell.cell_type == "code":
            print(f"[{model}] CELL: {cell.metadata.stage}", flush=True)

    try:
        client = ProgressClient(notebook, timeout=None, kernel_name="suim-unet-resnet18",
                                resources={"metadata": {"path": str(root)}}, on_cell_start=progress)
        client.execute()
        result = read_json(output / "summary.json")
        if smoke_folder is None:
            validate_pair(baseline, result)
            history = pd.read_csv(checkpoint / "training_history.csv")
            if history.epoch.tolist() != list(range(1, 51)):
                raise ValueError("Trial did not complete all 50 epochs")
            best = history.loc[history.val_loss.idxmin()]
            if (int(best.epoch) != result["checkpoint_metadata"]["epoch"]
                    or abs(best.val_loss - result["checkpoint_metadata"]["val_loss"]) > 1e-10):
                raise ValueError("Trial checkpoint and validation history disagree")
        else:
            assert result["epochs_requested"] == 1 and result["test_images_evaluated"] == 2
            assert result["train_pairs"] == result["validation_pairs"] == 2
            assert result["checkpoint_metadata"]["use_augmentation"] is False
        assert result["device"] == "cuda" and result["mode"] == "train"
        assert sha256(checkpoint / "best.pt") == result["weights_sha256"]
        notebook.metadata.augmentation_experiment.update(
            completed=True, elapsed_seconds=time.time() - started,
            epochs=result["epochs_requested"], evaluated_images=result["test_images_evaluated"])
        return result
    finally:
        nbformat.write(notebook, saved)


def collect_results(root=ROOT):
    """Reaggregate the same per-image metrics for both arms; positive delta favors augmentation."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_test_overlap as overlap

    root = Path(root).resolve()
    original_models = overlap.MODELS
    try:
        overlap.MODELS = [name for model in MODELS for name in (model, model + "_noaug")]
        audit, scores, categories = overlap.audit(root)
    finally:
        overlap.MODELS = original_models
    rows = []
    for model in MODELS:
        baseline = read_json(output_dir(root, model, augmented=True) / "summary.json")
        trial_path = output_dir(root, model, augmented=False) / "summary.json"
        trial = read_json(trial_path) if trial_path.is_file() else None
        if trial is not None:
            validate_pair(baseline, trial)
        for protocol in (PRIMARY, "original_test"):
            augmented = scores.loc[(scores.model == model) & (scores.protocol == protocol)].iloc[0]
            plain = scores.loc[(scores.model == model + "_noaug") & (scores.protocol == protocol)]
            row = {"model": model, "protocol": protocol, "test_images": int(augmented.test_images),
                   "augmented_iou_pct": float(augmented.iou_pct),
                   "augmented_boundary_f_pct": float(augmented.boundary_f_pct),
                   "noaug_iou_pct": None, "noaug_boundary_f_pct": None,
                   "augmentation_iou_gain_pp": None, "augmentation_boundary_f_gain_pp": None,
                   "augmented_best_epoch": baseline["checkpoint_metadata"]["epoch"],
                   "noaug_best_epoch": None, "status": "pending"}
            if len(plain) and trial is not None:
                value = plain.iloc[0]
                row.update(noaug_iou_pct=float(value.iou_pct),
                           noaug_boundary_f_pct=float(value.boundary_f_pct),
                           augmentation_iou_gain_pp=float(augmented.iou_pct - value.iou_pct),
                           augmentation_boundary_f_gain_pp=float(augmented.boundary_f_pct - value.boundary_f_pct),
                           noaug_best_epoch=trial["checkpoint_metadata"]["epoch"], status="complete")
            rows.append(row)
    folder = report_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(folder / "comparison.csv", index=False)
    scores.to_csv(folder / "all_scores.csv", index=False)
    categories.to_csv(folder / "class_metrics.csv", index=False)
    write_json(folder / "test_overlap_audit.json", audit)
    write_json(folder / "summary.json", {
        "experiment": "Same-seed existing geometric augmentation versus augmentation disabled",
        "primary_protocol": PRIMARY, "delta_definition": "augmented score minus no-augmentation score, percentage points",
        "completed_models": sorted(set(comparison.loc[comparison.status == "complete", "model"])),
        "planned_models": list(MODELS), "seed": 42, "epochs_per_model_per_condition": 50,
        "baseline_arm": "Reuse the completed augmented runs; no hyperparameters are changed using TEST",
        "limitations": "Single seed; tests existing flips/geometric policy only; excludes exact file overlap, not near-duplicates",
        "results": rows,
    })
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--report-only", action="store_true", help="Refresh the comparison without training")
    args = parser.parse_args()
    root = args.root.resolve()
    baselines = preflight(root)
    folder = report_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        protected = protected_hashes(root)
        write_json(folder / "protected_baseline_sha256.json", protected)
        for model in MODELS if args.model == "all" else [args.model]:
            existing = output_dir(root, model, augmented=False) / "summary.json"
            if existing.is_file():
                raise FileExistsError(f"{model}: trial already exists at {existing}; use --report-only to inspect it")
            try:
                write_json(folder / "status.json", {"state": "training", "model": model, "epochs": 50})
                run_trial(root, model, baselines[model])
                check_protected(root, protected)
                comparison = collect_results(root)
                print(comparison.loc[comparison.protocol == PRIMARY].round(3).to_string(index=False), flush=True)
            except Exception as error:
                write_json(folder / "status.json", {"state": "failed", "model": model, "error": str(error)})
                raise
        write_json(folder / "status.json", {"state": "complete", "completed_models": read_json(folder / "summary.json")["completed_models"]})
    else:
        comparison = collect_results(root)
        print(comparison.loc[comparison.protocol == PRIMARY].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
