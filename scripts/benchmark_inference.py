"""Compare the three trained SUIM models at batch 1, 320 x 240, FP32 on one GPU."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from augmentation_experiment import (
    MODELS, check_protected, preflight, protected_hashes, read_json, sha256, write_json,
)
from audit_test_overlap import category_means

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"unet_resnet18": "U-Net-ResNet18 (CNN)", "segformer_b0": "SegFormer-B0",
          "unetformer_resnet18": "UNetFormer-ResNet18 (hybrid)"}


def balanced_order(round_index):
    models = list(MODELS)
    offset = round_index % len(models)
    return models[offset:] + models[:offset]


def nvidia_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,"
             "memory.total,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi telemetry unavailable"


def make_inventory(root):
    root = Path(root)
    baselines = preflight(root)
    split = pd.read_csv(root / "data/test/output/unet_resnet18/split.csv", dtype=str)
    pool_hashes = set(split.image_sha256)
    images, excluded = [], []
    for path in sorted((root / "TEST/images").iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            record = {"stem": path.stem, "path": str(path.relative_to(root)), "sha256": sha256(path)}
            (excluded if record["sha256"] in pool_hashes else images).append(record)
    if len(images) != 106 or len(excluded) != 4:
        raise ValueError("Expected the shared 106-image subset and four exact TEST overlaps")
    accuracy = {}
    for model, summary in baselines.items():
        if summary["image_size_width_height"] != [320, 240] or summary["inference_precision"] != "float32":
            raise ValueError(f"{model}: unexpected inference settings")
        scores = pd.read_csv(root / "data/test/output" / model / "per_image_metrics.csv")
        selected = scores.loc[scores.stem.isin({r["stem"] for r in images})]
        if len(selected) != 530 or selected.duplicated(["stem", "class"]).any():
            raise ValueError(f"{model}: incomplete saved evaluation")
        means = category_means(selected)
        accuracy[model] = {"iou_pct": float(means.iou_pct.mean()),
                           "boundary_f_pct": float(means.boundary_f_pct.mean())}
    return {"images": images, "excluded_images": excluded, "baselines": baselines, "accuracy": accuracy}


def load_checkpoint_model(root, model_name, summary):
    """Execute only definitions/configuration and the existing strict checkpoint loader."""
    notebook = nbformat.read(Path(root) / "notebooks" / MODELS[model_name], as_version=4)
    sources = {c.metadata.stage: c.source for c in notebook.cells if c.cell_type == "code"}
    namespace = {"__name__": "suim_inference_benchmark"}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(Path(root) / ".cache/matplotlib")
    exec(compile(sources["imports"], "notebook_imports", "exec"), namespace)
    torch = namespace["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("Select the verified .venv-segformer CUDA environment")
    torch.set_num_threads(1)
    with redirect_stdout(io.StringIO()):
        exec(compile(sources["config"], "notebook_config", "exec"), namespace)
    namespace.update(MODE="checkpoint", DEVICE=torch.device("cuda:0"),
                     CUSTOM_WEIGHTS=Path(root) / "ckpt" / model_name / "best.pt",
                     SPLIT_SHA256=summary["split_sha256"])
    if sha256(namespace["CUSTOM_WEIGHTS"]) != summary["weights_sha256"]:
        raise ValueError("Checkpoint differs from the saved accuracy evaluation")
    used = ["imports", "config", "model_functions", "model_and_weights"]
    if "model_definition" in sources:
        exec(compile(sources["model_definition"], "notebook_model_definition", "exec"), namespace)
        used.append("model_definition")
    exec(compile(sources["model_functions"], "notebook_model_functions", "exec"), namespace)
    exec(compile(sources["model_and_weights"], "notebook_checkpoint_loading", "exec"), namespace)
    # Reuse the exact normalization function without constructing training loaders.
    node = next(node for node in ast.parse(sources["loader"]).body
                if isinstance(node, ast.FunctionDef) and node.name == "normalize_image")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_normalization", "exec"), namespace)
    model = namespace["model"].eval()
    print("Benchmark: saved checkpoint, FP32 inference, batch 1, 320 x 240", flush=True)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("All model parameters must be FP32")
    if namespace["CHECKPOINT_METADATA"] != summary["checkpoint_metadata"]:
        raise ValueError("Checkpoint metadata differs from its accuracy report")
    namespace["source_hashes"] = {stage: __import__("hashlib").sha256(sources[stage].encode()).hexdigest()
                                   for stage in [*used, "loader", "predict_functions"]}
    return namespace


def prepare_image(path, normalize):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((320, 240), Image.Resampling.BICUBIC)
    return normalize(image).unsqueeze(0)


def prediction(namespace, images):
    logits = namespace["full_resolution_logits"](namespace["model"], images, (240, 320))
    return logits.sigmoid() > 0.5


def verify_masks(actual, root, model_name, stem, classes):
    if actual.shape != (1, 5, 240, 320) or actual.dtype != np.bool_:
        raise ValueError("Expected five full-resolution binary output masks")
    for channel, category in enumerate(classes):
        with Image.open(Path(root) / "data/test/output" / model_name / category / f"{stem}.bmp") as source:
            expected = np.asarray(source) > 0
        if not np.array_equal(actual[0, channel], expected):
            raise ValueError(f"Benchmark prediction differs from saved results: {model_name}/{category}/{stem}")


def summarize_samples(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing samples must be finite positive milliseconds")
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": float(1000 * len(values) / values.sum()), "samples": int(len(values))}


def run_worker(root, folder, model_name, round_index, passes, warmup, smoke):
    root, folder = Path(root), Path(folder)
    inventory = read_json(folder / "inventory.json")
    summary = inventory["baselines"][model_name]
    records = inventory["images"][:2] if smoke else inventory["images"]
    start_telemetry = nvidia_snapshot()
    namespace = load_checkpoint_model(root, model_name, summary)
    torch = namespace["torch"]
    device = torch.device("cuda:0")
    samples = [prepare_image(root / r["path"], namespace["normalize_image"]) for r in records]
    model = namespace["model"]
    result_folder = folder / f"round_{round_index + 1}" / model_name
    result_folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        # Correctness and image loading are outside timing; this also warms the OS file cache.
        for record, sample in zip(records, samples):
            if sha256(root / record["path"]) != record["sha256"]:
                raise ValueError("TEST image changed after inventory")
            actual = prediction(namespace, sample.to(device)).cpu().numpy()
            verify_masks(actual, root, model_name, record["stem"], summary["classes"])
        del actual
        print(f"[{model_name}] round {round_index + 1}: verified {len(records)} saved predictions", flush=True)
        for index in range(warmup):
            prediction(namespace, samples[index % len(samples)].to(device))
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        end.record()
        end.synchronize()  # Initialize timing events before measurement.
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        measured = []
        for pass_index in range(passes):
            for record, sample in zip(records, samples):
                images = sample.to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                begin.record()
                binary = prediction(namespace, images)
                end.record()
                end.synchronize()
                elapsed = 1000 * (time.perf_counter() - started)
                measured.append({"model": model_name, "round": round_index + 1, "pass": pass_index + 1,
                                 "stem": record["stem"], "prediction_ms": elapsed,
                                 "gpu_event_ms": begin.elapsed_time(end)})
                del binary, images
        prediction_peak = torch.cuda.max_memory_allocated(device)
        reserved_peak = torch.cuda.max_memory_reserved(device)
        print(f"[{model_name}] round {round_index + 1}: GPU-ready prediction timing complete", flush=True)
        # Include file decode, resize, normalization, H2D and D2H. Exclude mask file saving.
        torch.cuda.reset_peak_memory_stats(device)
        pipeline = []
        for pass_index in range(passes):
            for record in records:
                torch.cuda.synchronize()
                started = time.perf_counter()
                images = prepare_image(root / record["path"], namespace["normalize_image"]).to(device)
                binary = prediction(namespace, images).cpu().numpy()
                elapsed = 1000 * (time.perf_counter() - started)
                pipeline.append({"model": model_name, "round": round_index + 1, "pass": pass_index + 1,
                                 "stem": record["stem"], "image_to_mask_ms": elapsed})
                del binary, images
        pipeline_peak = torch.cuda.max_memory_allocated(device)
    pd.DataFrame(measured).to_csv(result_folder / "prediction_samples.csv", index=False)
    pd.DataFrame(pipeline).to_csv(result_folder / "pipeline_samples.csv", index=False)
    info = {
        "model": model_name, "round": round_index + 1, "smoke": smoke,
        "test_images": len(records), "passes": passes, "warmup_iterations": warmup,
        "batch_size": 1, "image_size_width_height": [320, 240], "precision": "float32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "checkpoint_sha256": summary["weights_sha256"], "checkpoint_epoch": summary["checkpoint_metadata"]["epoch"],
        "verified_prediction_images": len(records), "source_code_sha256": namespace["source_hashes"],
        "prediction": summarize_samples([r["prediction_ms"] for r in measured]),
        "gpu_events": summarize_samples([r["gpu_event_ms"] for r in measured]),
        "image_to_mask": summarize_samples([r["image_to_mask_ms"] for r in pipeline]),
        "baseline_allocated_mib": baseline_allocated / 2**20,
        "prediction_peak_allocated_mib": prediction_peak / 2**20,
        "prediction_peak_reserved_mib": reserved_peak / 2**20,
        "pipeline_peak_allocated_mib": pipeline_peak / 2**20,
        "gpu": torch.cuda.get_device_name(device), "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_cpu_threads": torch.get_num_threads(), "python": platform.python_version(),
        "platform": platform.platform(), "gpu_telemetry_before": start_telemetry,
        "gpu_telemetry_after": nvidia_snapshot(),
        "packages": {name: version(name) for name in ("torchvision", "transformers", "timm",
                     "segmentation-models-pytorch", "numpy", "Pillow")},
    }
    write_json(result_folder / "result.json", info)
    print(f"[{model_name}] round {round_index + 1}: {info['prediction']['mean_ms']:.2f} ms prediction, "
          f"{info['image_to_mask']['mean_ms']:.2f} ms image-to-mask", flush=True)


def aggregate(folder, rounds, passes, smoke):
    folder = Path(folder)
    inventory = read_json(folder / "inventory.json")
    results, raw_prediction, raw_pipeline = [], [], []
    for model in MODELS:
        runs = [read_json(folder / f"round_{r + 1}" / model / "result.json") for r in range(rounds)]
        for key in ("checkpoint_sha256", "parameters", "gpu", "precision", "batch_size", "torch",
                    "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32", "matmul_allow_tf32"):
            if len({json.dumps(run[key], sort_keys=True) for run in runs}) != 1:
                raise ValueError(f"{model}: inconsistent {key} across rounds")
        prediction_samples = pd.concat([pd.read_csv(folder / f"round_{r + 1}" / model / "prediction_samples.csv")
                                        for r in range(rounds)], ignore_index=True)
        pipeline_samples = pd.concat([pd.read_csv(folder / f"round_{r + 1}" / model / "pipeline_samples.csv")
                                      for r in range(rounds)], ignore_index=True)
        expected_images = 2 if smoke else 106
        expected_stems = {r["stem"] for r in inventory["images"][:expected_images]}
        for table in (prediction_samples, pipeline_samples):
            if (len(table) != rounds * passes * expected_images or set(table.stem) != expected_stems
                    or table.duplicated(["round", "pass", "stem"]).any()):
                raise ValueError(f"{model}: incomplete timing sample grid")
        row = {"model": model, "label": LABELS[model], "parameters": runs[0]["parameters"],
               "parameters_million": runs[0]["parameters"] / 1e6, "rounds": rounds,
               "samples_per_phase": len(prediction_samples), "batch_size": 1, "precision": "float32",
               "test_images": expected_images, "gpu": runs[0]["gpu"],
               "peak_allocated_mib": max(run["prediction_peak_allocated_mib"] for run in runs),
               "peak_reserved_mib": max(run["prediction_peak_reserved_mib"] for run in runs),
               "pipeline_peak_allocated_mib": max(run["pipeline_peak_allocated_mib"] for run in runs),
               "round_prediction_mean_std_ms": float(np.std([run["prediction"]["mean_ms"] for run in runs], ddof=1)) if rounds > 1 else 0,
               "round_image_to_mask_mean_std_ms": float(np.std([run["image_to_mask"]["mean_ms"] for run in runs], ddof=1)) if rounds > 1 else 0,
               **inventory["accuracy"][model]}
        for prefix, column, table in (("prediction", "prediction_ms", prediction_samples),
                                      ("gpu_event", "gpu_event_ms", prediction_samples),
                                      ("image_to_mask", "image_to_mask_ms", pipeline_samples)):
            row.update({f"{prefix}_{key}": value for key, value in summarize_samples(table[column]).items()})
        results.append(row)
        raw_prediction.append(prediction_samples)
        raw_pipeline.append(pipeline_samples)
    table = pd.DataFrame(results)
    table.to_csv(folder / "comparison.csv", index=False)
    pd.concat(raw_prediction, ignore_index=True).to_csv(folder / "prediction_samples.csv", index=False)
    pd.concat(raw_pipeline, ignore_index=True).to_csv(folder / "pipeline_samples.csv", index=False)
    write_json(folder / "summary.json", {
        "completed": True, "smoke": smoke, "rounds": rounds, "passes_per_round": passes,
        "images_per_pass": expected_images, "results": results,
        "prediction_scope": "GPU-resident input to five binary GPU masks; synchronized wall time including dispatch and event instrumentation",
        "gpu_event_scope": "Same prediction between synchronized CUDA events; excludes input transfer",
        "image_to_mask_scope": "Warm-file-cache decode, bicubic resize, ImageNet normalization, H2D, prediction, D2H; no mask encoding/writes",
        "fps_definition": "1000 divided by mean latency in milliseconds, batch 1; not an overlapping or batched throughput test",
        "memory_scope": "Peak PyTorch tensor allocation in a fresh single-model worker; includes weights/input/activations/output; excludes CUDA context/other processes",
        "precision_scope": "FP32 parameters and inputs, autocast disabled, notebook TF32 defaults retained and recorded",
        "limitations": "Local laptop GPU, warm start and warm file cache; dynamic clocks/background load can vary; eager implementations, no export/compile/FP16 optimization",
    })
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="Two-image GPU check in a separate smoke folder")
    parser.add_argument("--worker-model", choices=list(MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--round-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.passes, args.warmup) < 1:
        parser.error("rounds, passes and warmup must be positive")
    root = args.root.resolve()
    folder = root / "artifacts/speed_comparison"
    if args.smoke:
        folder = folder / "smoke"
    if args.worker_model:
        run_worker(root, folder, args.worker_model, args.round_index, args.passes, args.warmup, args.smoke)
        return
    inventory = make_inventory(root)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "inventory.json", inventory)
    protected = protected_hashes(root)
    write_json(folder / "protected_baseline_sha256.json", protected)
    rounds, passes, warmup = (1, 1, 3) if args.smoke else (args.rounds, args.passes, args.warmup)
    started = time.time()
    try:
        for round_index in range(rounds):
            for model in balanced_order(round_index):
                write_json(folder / "status.json", {"state": "benchmarking", "model": model, "round": round_index + 1,
                                                    "rounds": rounds, "started_unix": started})
                command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                           "--worker-model", model, "--round-index", str(round_index),
                           "--passes", str(passes), "--warmup", str(warmup)]
                if args.smoke:
                    command.append("--smoke")
                subprocess.run(command, cwd=root, check=True,
                               env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        check_protected(root, protected)
        table = aggregate(folder, rounds, passes, args.smoke)
        write_json(folder / "status.json", {"state": "complete", "elapsed_seconds": time.time() - started,
                                            "baseline_files_unchanged": True, "smoke": args.smoke})
        print(table[["model", "parameters_million", "prediction_mean_ms", "prediction_fps",
                     "image_to_mask_fps", "peak_allocated_mib", "iou_pct"]].round(3).to_string(index=False))
    except Exception as error:
        write_json(folder / "status.json", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
