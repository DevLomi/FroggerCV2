"""Download and merge pothole datasets, fine-tune YOLO, and install a better validated model."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET_ID = "keremberke/pothole-segmentation"
EXTRA_DATASET_ID = "Ryukijano/Pothole-detection-Yolov8"
DEFAULT_DATASETS = (DATASET_ID, EXTRA_DATASET_ID)
DATA_DIR = ROOT / "data" / "pothole"
NEGATIVES_DIR = ROOT / "data" / "negatives"
MODEL_PATH = ROOT / "models" / "pothole.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _normalize_objects(objects, field):
    if isinstance(objects, dict):
        if len(objects["category"]) != len(objects[field]):
            raise ValueError(f"Annotation categories and {field} have different lengths.")
        return [{"category": c, field: v} for c, v in zip(objects["category"], objects[field])]
    return list(objects)


def segmentation_labels(objects, width, height):
    """Convert COCO pixel polygons to one normalized YOLO row per instance."""
    objects = _normalize_objects(objects, "segmentation")
    lines = []
    for obj in objects:
        if obj["category"] not in (0, "pothole", "Pothole"):
            raise ValueError(f"Unexpected category: {obj['category']!r}")
        polygons = obj["segmentation"]
        if not isinstance(polygons, list) or not polygons:
            raise ValueError("Each pothole must have polygon annotations.")
        validated = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
            if len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("A polygon requires at least three finite points.")
            points = np.clip(points / [width, height], 0, 1)
            area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1))
                       - np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
            if area <= 0:
                raise ValueError("A polygon has no area after normalization.")
            validated.append(points)
        if len(validated) > 1:
            # Match Ultralytics' COCO conversion: retain one object for multipart masks.
            from ultralytics.data.converter import merge_multi_segment
            points = np.concatenate(merge_multi_segment(validated), axis=0)
        else:
            points = validated[0]
        lines.append("0 " + " ".join(f"{value:.8f}" for value in points.ravel()))
    return "\n".join(lines) + ("\n" if lines else "")


def box_labels(objects, width, height):
    """Convert COCO pixel [x, y, w, h] bounding boxes to normalized YOLO rows."""
    objects = _normalize_objects(objects, "bbox")
    lines = []
    for obj in objects:
        if obj["category"] not in (0, "pothole", "Pothole"):
            raise ValueError(f"Unexpected category: {obj['category']!r}")
        box = np.asarray(obj["bbox"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Each bbox requires four finite numbers [x, y, w, h].")
        x, y, bw, bh = box
        if bw <= 0 or bh <= 0:
            raise ValueError("Bounding box width and height must be positive.")
        x1, y1 = np.clip(x, 0, width), np.clip(y, 0, height)
        x2, y2 = np.clip(x + bw, 0, width), np.clip(y + bh, 0, height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box has no area inside image bounds.")
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        nw, nh = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def yolo_text_labels(text):
    """Validate and normalize raw YOLO annotation text (boxes or polygons) to class 0 boxes."""
    if text is None or not str(text).strip():
        return ""
    lines = []
    for raw_line in str(text).strip().splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        cls_val = float(parts[0])
        if not np.isfinite(cls_val) or int(cls_val) != 0 or cls_val != 0:
            raise ValueError(f"Unexpected class label in YOLO text: {parts[0]!r}")
        coords = np.asarray(list(map(float, parts[1:])), dtype=float)
        if not np.isfinite(coords).all():
            raise ValueError("Non-finite coordinate in YOLO label.")
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            if bw <= 0 or bh <= 0 or cx < -0.01 or cy < -0.01 or cx > 1.01 or cy > 1.01:
                raise ValueError("Out-of-bounds YOLO box.")
            x1, y1 = np.clip(cx - bw / 2, 0, 1), np.clip(cy - bh / 2, 0, 1)
            x2, y2 = np.clip(cx + bw / 2, 0, 1), np.clip(cy + bh / 2, 0, 1)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.clip(coords.reshape(-1, 2), 0, 1)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
        else:
            raise ValueError(f"Unsupported YOLO row column count: {len(parts)}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Degenerate YOLO annotation after clipping.")
        lines.append(f"0 {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def row_labels(row, width, height):
    """Extract YOLO label lines from polygon objects, bbox objects, YOLO text, or empty negatives."""
    if "objects" in row and row["objects"] is not None:
        objects = row["objects"]
        if isinstance(objects, dict):
            if "segmentation" in objects and objects["segmentation"]:
                return segmentation_labels(objects, width, height)
            if "bbox" in objects:
                return box_labels(objects, width, height)
            return ""
        if isinstance(objects, list):
            if not objects:
                return ""
            if "segmentation" in objects[0]:
                return segmentation_labels(objects, width, height)
            if "bbox" in objects[0]:
                return box_labels(objects, width, height)
    if "text" in row:
        return yolo_text_labels(row["text"])
    if row.get("negative"):
        return ""
    raise ValueError("Row does not contain supported 'objects' or 'text' annotations.")


def export_dataset(datasets, destination=DATA_DIR, negatives_dir=None, dataset_ids=DEFAULT_DATASETS):
    """Merge one or more split-partitioned datasets and optional negative road frames without split leakage."""
    destination = Path(destination).resolve()
    if isinstance(datasets, dict):
        datasets = [datasets]
    counts = {}
    negatives_added = 0
    for source, target in (("train", "train"), ("validation", "val"), ("test", "test")):
        images = destination / "images" / target
        labels = destination / "labels" / target
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (destination / "labels" / f"{target}.cache").unlink(missing_ok=True)
        paths = []
        serial = 0
        for ds in datasets:
            split_key = source if source in ds else ("val" if source == "validation" and "val" in ds else None)
            if split_key is None:
                continue
            for index, row in enumerate(ds[split_key]):
                if "image" in row and row["image"] is not None:
                    image = row["image"].convert("RGB")
                elif "image_path" in row and row["image_path"] is not None:
                    with Image.open(row["image_path"]) as raw_img:
                        image = raw_img.convert("RGB")
                else:
                    raise ValueError(f"Row is missing 'image' or 'image_path': {source}/{index}")
                width, height = image.size
                if "width" in row and "height" in row and row["width"] and row["height"]:
                    if (width, height) != (row["width"], row["height"]):
                        raise ValueError(f"Image dimensions disagree with annotations: {source}/{index}")
                label = row_labels(row, width, height)
                name = f"{serial:06d}"
                serial += 1
                image_path = images / f"{name}.jpg"
                image.save(image_path, quality=95)
                (labels / f"{name}.txt").write_text(label, encoding="utf-8")
                paths.append(image_path.as_posix())
        if target == "train" and negatives_dir is not None and Path(negatives_dir).is_dir():
            neg_files = sorted(
                p for p in Path(negatives_dir).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for neg_path in neg_files:
                with Image.open(neg_path) as neg_img:
                    rgb = neg_img.convert("RGB")
                name = f"neg_{negatives_added:06d}"
                negatives_added += 1
                image_path = images / f"{name}.jpg"
                rgb.save(image_path, quality=95)
                (labels / f"{name}.txt").write_text("", encoding="utf-8")
                paths.append(image_path.as_posix())
        if not paths:
            raise ValueError(f"The {source} split is empty.")
        # Explicit lists prevent leftover files from entering a later training run.
        (destination / f"{target}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        counts[target] = len(paths)
    config = dict(path=destination.as_posix(), train="train.txt", val="val.txt",
                  test="test.txt", names={0: "pothole"})
    config_path = destination / "data.yaml"
    # JSON is valid YAML, so no extra serializer is needed.
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    sources_text = "\n".join(f"Source: https://huggingface.co/datasets/{ds_id}" for ds_id in dataset_ids)
    (destination / "SOURCE.txt").write_text(
        f"{sources_text}\nNegative frames added: {negatives_added}\n",
        encoding="utf-8",
    )
    print(f"Prepared {counts} (negatives={negatives_added}) at {destination}")
    return config_path


def load_yolo_folder_dataset(root_dir):
    """Load a YOLO-format directory (<split>/images + <split>/labels) into split row dicts."""
    root = Path(root_dir)
    splits = {}
    split_aliases = (
        ("train", ("train",)),
        ("validation", ("validation", "valid", "val")),
        ("test", ("test",)),
    )
    for target, candidates in split_aliases:
        images_dir = next(
            (p for c in candidates for p in (root / c / "images", root / "images" / c) if p.is_dir()),
            None,
        )
        if images_dir is None:
            continue
        labels_dir = (
            images_dir.parent / "labels"
            if images_dir.name == "images"
            else root / "labels" / images_dir.name
        )
        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        rows = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
            rows.append({"image_path": img_path, "text": text})
        if rows:
            splits[target] = rows
    if not splits:
        raise ValueError(f"No YOLO split folders (<split>/images) found in {root}")
    return splits


def load_hf_dataset(dataset_id, cache_dir):
    """Load a Hugging Face dataset via Parquet shards, YOLO folder snapshot, or standard builder."""
    local_dir = Path(dataset_id)
    if local_dir.is_dir():
        return load_yolo_folder_dataset(local_dir)

    from datasets import load_dataset

    if dataset_id == DATASET_ID:
        base = f"https://huggingface.co/datasets/{dataset_id}/resolve/refs%2Fconvert%2Fparquet/full"
        files = {split: f"{base}/{split}/0000.parquet" for split in ("train", "validation", "test")}
        return load_dataset("parquet", data_files=files, cache_dir=cache_dir)

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(snapshot_download(dataset_id, repo_type="dataset", cache_dir=cache_dir))
    if any(
        p.is_dir()
        for c in ("train", "validation", "valid", "val", "test")
        for p in (snapshot_dir / c / "images", snapshot_dir / "images" / c)
    ):
        return load_yolo_folder_dataset(snapshot_dir)
    return load_dataset(dataset_id, cache_dir=cache_dir)


def prepare_dataset(dataset_ids=DEFAULT_DATASETS, negatives_dir=NEGATIVES_DIR):
    cache_dir = str(ROOT / "data" / "hf_cache")
    loaded = [load_hf_dataset(ds_id, cache_dir) for ds_id in dataset_ids]
    return export_dataset(loaded, destination=DATA_DIR, negatives_dir=negatives_dir, dataset_ids=dataset_ids)


def train(args):
    dataset_ids = tuple(getattr(args, "datasets", None) or DEFAULT_DATASETS)
    negatives_dir = getattr(args, "negatives_dir", NEGATIVES_DIR)
    data = prepare_dataset(dataset_ids=dataset_ids, negatives_dir=negatives_dir)
    if args.prepare_only:
        return
    from ultralytics import YOLO
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    weights = args.weights or str(MODEL_PATH if MODEL_PATH.is_file() else MODEL_PATH.parent / "yolov8n-seg.pt")
    model = YOLO(weights)
    if model.task not in ("detect", "segment"):
        raise ValueError("Training requires a detection or segmentation checkpoint.")
    project_dir = ROOT / "runs" / model.task
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, seed=42, patience=args.patience,
        optimizer="AdamW", lr0=args.lr, lrf=0.1, cos_lr=True, warmup_epochs=3,
        mosaic=0.2, close_mosaic=10, scale=0.25, hsv_s=0.4, hsv_v=0.3,
        project=str(project_dir), name="pothole", exist_ok=False,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt; the active model was not changed.")
    install_if_better(best, data, args)


def install_if_better(best, data, args):
    """Select on validation data; keep the active checkpoint if the candidate regresses."""
    from ultralytics import YOLO

    settings = dict(data=str(data), imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=0, project=str(ROOT / "runs" / "eval"))
    candidate = YOLO(str(best))
    validation = candidate.val(**settings, split="val", name="candidate_validation")
    # Box mAP is comparable across both detection and segmentation checkpoints.
    candidate_score = float(validation.box.map)
    if not np.isfinite(candidate_score):
        raise RuntimeError("Invalid candidate validation score; the active model was not changed.")
    if MODEL_PATH.is_file():
        current = YOLO(str(MODEL_PATH)).val(**settings, split="val", name="current_validation")
        current_score = float(current.box.map)
        if not np.isfinite(current_score):
            raise RuntimeError("Invalid current validation score; the active model was not changed.")
        print(f"Validation box mAP50-95: current={current_score:.4f}, candidate={candidate_score:.4f}")
        if candidate_score <= current_score:
            print(f"Kept the current model. Candidate checkpoint: {best}")
            return False
    # Test images report generalization only; they do not choose the winner.
    metrics = candidate.val(**settings, split="test", name="pothole_test")
    print(f"Test-split box mAP50-95: {metrics.box.map:.4f}")
    pending = MODEL_PATH.with_suffix(".pt.tmp")
    shutil.copy2(best, pending)
    if MODEL_PATH.is_file():
        shutil.copy2(MODEL_PATH, MODEL_PATH.with_name("pothole.previous.pt"))
    pending.replace(MODEL_PATH)
    print(f"Installed trained model: {MODEL_PATH}\nRun python detection.py to use it.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Use 0 for the first CUDA GPU, or cpu.")
    parser.add_argument("--weights", help="Start from a checkpoint, e.g. models/pothole.pt, to fine-tune it.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Initial AdamW learning rate.")
    parser.add_argument("--patience", type=int, default=25, help="Epochs without validation improvement; 0 disables early stopping.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        help="Hugging Face dataset IDs to merge across splits.")
    parser.add_argument("--negatives-dir", type=Path, default=NEGATIVES_DIR,
                        help="Folder of negative/background road images (empty labels).")
    parser.add_argument("--prepare-only", action="store_true", help="Download and convert without training.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("epochs and batch must be positive; imgsz must be a positive multiple of 32.")
    if not np.isfinite(args.lr) or args.lr <= 0 or args.patience < 0:
        parser.error("lr must be positive and finite; patience must be nonnegative.")
    train(args)


if __name__ == "__main__":
    main()
