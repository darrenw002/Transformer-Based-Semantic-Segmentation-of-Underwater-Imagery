"""Export reviewed SUIM crops and optional class masks without changing inputs.

Standalone requirements: Python >= 3.10, NumPy, Pillow. Run with --help.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


# Embedded so this script can be shared with someone who only has the dataset.
_SMALL_CROPS = """
f_r_1058_ f_r_1068_ f_r_1069_ f_r_1070_ f_r_1133_ f_r_1142_ f_r_1151_
f_r_1154_ f_r_1214_ f_r_1233_ f_r_1289_ f_r_1302_ f_r_1318_ f_r_1424_
f_r_1491_ f_r_1515_ f_r_1779_ f_r_1816_ f_r_1866_ f_r_1879_ f_r_829_
f_r_921_ f_r_934_ f_r_968_ f_r_991_ w_r_1_ w_r_24_ w_r_25_ w_r_27_
w_r_47_ w_r_7_
""".split()
_LARGE_CROPS = "f_r_1259_ f_r_1290_ f_r_1394_ f_r_1570_ f_r_1812_ f_r_401_".split()
REVIEWED_CROPS = {
    **{stem: ((590, 375), (590, 430)) for stem in _SMALL_CROPS},
    **{stem: ((910, 435), (910, 490)) for stem in _LARGE_CROPS},
}
SYMBOLS = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")
LABELS = ("Water", "Divers", "Plants", "Wrecks", "Robots", "Reefs", "Fish", "Sea-floor")
PALETTE = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
           (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
COLOR_POLICIES = ("preserve", "ignore", "threshold")


@dataclass
class Sample:
    stem: str
    image: Path
    mask: Path
    image_size: tuple[int, int]
    mask_size: tuple[int, int]
    crop_rows: int
    image_sha256: str
    mask_sha256: str
    aligned_rgb_sha256: str
    noncanonical_pixels: int
    canonical_counts: list[int]
    decoded_counts: list[int]
    keep: bool = True


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_hash(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def index_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in files:
                raise ValueError(f"Ambiguous filename stem: {folder / path.stem}")
            files[path.stem] = path
    return files


def inventory(folder: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images = index_files(folder / "images", {".jpg", ".jpeg", ".png"})
    masks = index_files(folder / "masks", {".bmp", ".png"})
    if images.keys() != masks.keys():
        raise ValueError(
            f"Unpaired files in {folder}: missing masks={sorted(images.keys() - masks.keys())[:5]}, "
            f"missing images={sorted(masks.keys() - images.keys())[:5]}"
        )
    return images, masks


def aligned_rgb(stem: str, image_size: tuple[int, int], mask: Path) -> tuple[np.ndarray, tuple[int, int], int]:
    with Image.open(mask) as source:
        original_size = source.size
        rgb = source.convert("RGB")
    reviewed = REVIEWED_CROPS.get(stem)
    if reviewed is not None and image_size != reviewed[0]:
        raise ValueError(f"{stem}: reviewed image dimensions changed: {image_size}, expected {reviewed[0]}")
    if image_size == original_size:
        return np.array(rgb), original_size, 0
    if reviewed != (image_size, original_size):
        raise ValueError(f"{stem}: unreviewed geometry: image={image_size}, mask={original_size}")
    return np.array(rgb.crop((0, 0, *image_size))), original_size, 55


def decode(rgb: np.ndarray, policy: str) -> tuple[np.ndarray | None, np.ndarray]:
    exact = ((rgb == 0) | (rgb == 255)).all(axis=-1)
    if policy == "preserve":
        return None, exact
    bits = (rgb >= 128).astype(np.uint8)
    ids = 4 * bits[..., 0] + 2 * bits[..., 1] + bits[..., 2]
    if policy == "ignore":
        ids[~exact] = 255
    return ids, exact


def scan(source: Path, color_policy: str) -> tuple[list[Sample], int | None]:
    images, masks = inventory(source / "train_val")
    if not images:
        raise ValueError("No training image-mask pairs found")
    # TEST is inventoried by filenames only; its pixels are never read.
    test_pairs = len(inventory(source / "TEST")[0]) if (source / "TEST").exists() else None
    samples = []
    for index, (stem, image_path) in enumerate(images.items(), start=1):
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        rgb, mask_size, rows = aligned_rgb(stem, image_size, masks[stem])
        ids, exact = decode(rgb, color_policy)
        bits = (rgb == 255).astype(np.uint8)
        canonical_ids = 4 * bits[..., 0] + 2 * bits[..., 1] + bits[..., 2]
        counts = np.bincount(canonical_ids[exact], minlength=8).tolist()
        decoded = np.bincount(ids[ids != 255], minlength=8).tolist() if ids is not None else []
        samples.append(Sample(
            stem, image_path, masks[stem], image_size, mask_size, rows,
            file_hash(image_path), file_hash(masks[stem]), rgb_hash(rgb),
            int((~exact).sum()), counts, decoded,
        ))
        if index % 250 == 0:
            print(f"Scanned {index}/{len(images)} pairs", flush=True)
    return samples, test_pairs


def duplicate_groups(samples: list[Sample], decisions: Path | None) -> list[dict]:
    by_hash = {}
    for sample in samples:
        by_hash.setdefault(sample.image_sha256, []).append(sample)
    groups = {digest: members for digest, members in by_hash.items() if len(members) > 1}
    selections = {}
    if decisions is not None:
        with decisions.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"image_sha256", "keep_stem"}.issubset(reader.fieldnames or []):
                raise ValueError("Duplicate decisions require image_sha256 and keep_stem columns")
            for row in reader:
                if row["image_sha256"] is None or row["keep_stem"] is None:
                    raise ValueError("Duplicate decision row is missing image_sha256 or keep_stem")
                digest, keep = row["image_sha256"].strip(), row["keep_stem"].strip()
                if digest not in groups:
                    raise ValueError(f"Duplicate decision hash does not match a current group: {digest}")
                if digest in selections:
                    raise ValueError(f"Repeated duplicate decision: {digest}")
                if keep and keep not in {sample.stem for sample in groups[digest]}:
                    raise ValueError(f"keep_stem {keep!r} is not in duplicate group {digest}")
                selections[digest] = keep
    rows = []
    for digest, members in sorted(groups.items()):
        keep = selections.get(digest, "")
        if keep:
            for sample in members:
                sample.keep = sample.stem == keep
        masks_identical = len({(s.image_size, s.aligned_rgb_sha256) for s in members}) == 1
        rows.append({
            "image_sha256": digest, "stems": ";".join(s.stem for s in members),
            "rgb_masks_identical": masks_identical, "keep_stem": keep,
            "decision": "selected" if keep else "kept_all",
        })
    return rows


def validate_output(source: Path, output: Path | None, dry_run: bool) -> None:
    if output is None:
        if not dry_run:
            raise ValueError("--output is required unless --dry-run is used")
        return
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if output == source or output in source.parents:
        raise ValueError("Output cannot be the source directory or one of its parents")
    for protected in (source / "train_val", source / "TEST"):
        if output == protected or protected in output.parents:
            raise ValueError(f"Output cannot be inside source data: {protected}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_sample(sample: Sample, output: Path, color_policy: str) -> dict[str, str]:
    if file_hash(sample.mask) != sample.mask_sha256:
        raise ValueError(f"Source mask changed during export: {sample.mask}")
    rgb, _, _ = aligned_rgb(sample.stem, sample.image_size, sample.mask)
    if rgb_hash(rgb) != sample.aligned_rgb_sha256:
        raise ValueError(f"Source mask pixels changed during export: {sample.mask}")
    image_path = output / "train_val/images" / sample.image.name
    mask_path = output / "train_val/masks" / f"{sample.stem}.png"
    shutil.copyfile(sample.image, image_path)
    output_image_hash = file_hash(image_path)
    if output_image_hash != sample.image_sha256:
        raise ValueError(f"Source image changed during export: {sample.image}")
    Image.fromarray(rgb).save(mask_path)
    with Image.open(mask_path) as saved:
        if rgb_hash(np.array(saved.convert("RGB"))) != sample.aligned_rgb_sha256:
            raise ValueError(f"Exported mask failed pixel preservation: {mask_path}")
    result = {
        "output_image": image_path.relative_to(output).as_posix(),
        "output_image_sha256": output_image_hash,
        "output_mask": mask_path.relative_to(output).as_posix(),
        "output_mask_sha256": file_hash(mask_path),
        "output_class_mask": "", "output_class_mask_sha256": "",
    }
    ids, _ = decode(rgb, color_policy)
    if ids is not None:
        class_path = output / "train_val/masks_class_id" / f"{sample.stem}.png"
        Image.fromarray(ids).save(class_path)
        with Image.open(class_path) as saved:
            if not np.array_equal(np.array(saved), ids):
                raise ValueError(f"Exported class mask failed pixel preservation: {class_path}")
        result.update(output_class_mask=class_path.relative_to(output).as_posix(),
                      output_class_mask_sha256=file_hash(class_path))
    return result


def clean_dataset(source: Path, output: Path | None = None, *, color_policy: str = "preserve",
                  decisions: Path | None = None, dry_run: bool = False) -> dict:
    if color_policy not in COLOR_POLICIES:
        raise ValueError(f"Unknown color policy: {color_policy}")
    source = source.expanduser().resolve()
    output = output.expanduser().resolve() if output is not None else None
    validate_output(source, output, dry_run)
    samples, test_pairs = scan(source, color_policy)
    groups = duplicate_groups(samples, decisions)
    kept = [sample for sample in samples if sample.keep]
    summary = {
        "schema_version": 1,
        "source": str(source), "output": str(output) if output else None,
        "dry_run": dry_run, "color_policy": color_policy,
        "input_pairs": len(samples), "reviewed_crops_applied": sum(s.crop_rows > 0 for s in samples),
        "reviewed_pairs_already_aligned": sum(s.stem in REVIEWED_CROPS and not s.crop_rows for s in samples),
        "input_pairs_with_noncanonical_colors": sum(s.noncanonical_pixels > 0 for s in samples),
        "input_noncanonical_pixels_after_crop": sum(s.noncanonical_pixels for s in samples),
        "duplicate_image_groups": len(groups),
        "duplicate_groups_with_different_masks": sum(not g["rgb_masks_identical"] for g in groups),
        "duplicate_groups_kept_all": sum(not g["keep_stem"] for g in groups),
        "excluded_pairs": len(samples) - len(kept), "output_pairs": len(kept),
        "output_rgb_noncanonical_pixels": sum(s.noncanonical_pixels for s in kept),
        "output_class_mask_counts": {symbol: sum(s.decoded_counts[i] for s in kept)
                                     for i, symbol in enumerate(SYMBOLS)} if color_policy != "preserve" else None,
        "output_ignored_pixels": sum(s.noncanonical_pixels for s in kept) if color_policy == "ignore" else 0,
        "test_inventory_pairs": test_pairs, "test_exported": False,
    }
    if dry_run:
        return summary

    # Validate all input geometry and manual decisions before creating outputs.
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Export in progress or failed. Do not train on this directory.\n", encoding="utf-8")
    for folder in ("train_val/images", "train_val/masks", "reports"):
        (output / folder).mkdir(parents=True)
    if color_policy != "preserve":
        (output / "train_val/masks_class_id").mkdir()
    duplicate_hashes = {group["image_sha256"] for group in groups}
    rows = []
    for index, sample in enumerate(samples, start=1):
        row = {
            "stem": sample.stem, "retained": sample.keep,
            "source_image": sample.image.relative_to(source).as_posix(),
            "source_mask": sample.mask.relative_to(source).as_posix(),
            "image_width": sample.image_size[0], "image_height": sample.image_size[1],
            "source_mask_width": sample.mask_size[0], "source_mask_height": sample.mask_size[1],
            "bottom_rows_cropped": sample.crop_rows,
            "crop_box": f"0;0;{sample.image_size[0]};{sample.image_size[1]}" if sample.crop_rows else "",
            "noncanonical_pixels_after_crop": sample.noncanonical_pixels,
            "source_image_sha256": sample.image_sha256, "source_mask_sha256": sample.mask_sha256,
            "aligned_rgb_sha256": sample.aligned_rgb_sha256,
            "duplicate_group": sample.image_sha256 if sample.image_sha256 in duplicate_hashes else "",
            "output_image": "", "output_image_sha256": "", "output_mask": "", "output_mask_sha256": "",
            "output_class_mask": "", "output_class_mask_sha256": "",
        }
        row.update({f"exact_pixels_{symbol}": sample.canonical_counts[i] for i, symbol in enumerate(SYMBOLS)})
        if sample.keep:
            row.update(export_sample(sample, output, color_policy))
        rows.append(row)
        if index % 250 == 0:
            print(f"Exported {index}/{len(samples)} input records", flush=True)
    write_csv(output / "reports/pairs.csv", rows, list(rows[0]))
    write_csv(output / "reports/duplicate_groups.csv", groups,
              ["image_sha256", "stems", "rgb_masks_identical", "keep_stem", "decision"])
    definition = {
        "classes": [{"id": i, "symbol": symbol, "label": LABELS[i], "rgb": PALETTE[i]}
                    for i, symbol in enumerate(SYMBOLS)],
        "color_policy": color_policy, "ignore_id": 255 if color_policy == "ignore" else None,
        "threshold": 128 if color_policy == "threshold" else None,
        "rgb_masks": "Aligned source RGB values preserved, including noncanonical colors",
    }
    (output / "reports/class_definition.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    (output / "reports/summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# SUIM cleanup export\n\n"
        "train_val/images contains byte-identical copies of retained source images.\n"
        "train_val/masks contains aligned RGB PNG masks; source label colors are preserved.\n"
        "Optional masks_class_id PNGs contain class IDs, not RGB annotations.\n"
        f"Color policy: {color_policy}. Consult reports/class_definition.json.\n"
        "Ignore ID 255, if present, must be excluded from loss and metrics.\n"
        "Threshold decoding is a color mapping, not a verified semantic repair.\n"
        "TEST was inventoried only and is not copied here; use the unchanged original TEST split.\n"
        "reports/pairs.csv logs all source pairs and exclusions.\n"
        "reports/duplicate_groups.csv is also a review template: fill keep_stem, then pass it\n"
        "with --duplicate-decisions when exporting to a new output directory.\n"
        "Blank keep_stem retains all group members. Keep each group in one data partition.\n"
        "Read reports/summary.json for unresolved colors and retained duplicates.\n",
        encoding="utf-8",
    )
    incomplete.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="Original SUIM root (default: current directory)")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print the planned summary without writing files")
    parser.add_argument("--color-policy", choices=COLOR_POLICIES, default="preserve",
                        help="preserve: RGB only (default); ignore: class IDs with unknown=255; threshold: class IDs using RGB >= 128")
    parser.add_argument("--duplicate-decisions", type=Path,
                        help="Reviewed CSV with image_sha256 and keep_stem; unspecified groups keep all pairs")
    args = parser.parse_args()
    try:
        summary = clean_dataset(args.source, args.output, color_policy=args.color_policy,
                                decisions=args.duplicate_decisions, dry_run=args.dry_run)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
