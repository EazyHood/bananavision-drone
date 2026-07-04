from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .pipeline import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class SplitSummary:
    images: int
    groups: int
    train: int
    val: int
    test: int
    missing_labels: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitItem:
    image: Path
    label: Path
    relative: Path
    group: str


def split_yolo_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    output_root: str | Path,
    manifest_csv: str | Path | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 7,
    class_names: list[str] | None = None,
    write_empty_labels: bool = True,
) -> SplitSummary:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_root = Path(output_root)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
    ratios = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    group_lookup = _read_group_manifest(manifest_csv)
    items = _collect_items(image_dir, label_dir, group_lookup)
    groups: dict[str, list[SplitItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)
    assignments = _assign_groups(groups, ratios, seed)
    missing_labels = 0
    counts = {"train": 0, "val": 0, "test": 0}
    for split, split_items in assignments.items():
        for item in split_items:
            _copy_item(item, output_root, split, write_empty_labels=write_empty_labels)
            if not item.label.exists():
                missing_labels += 1
            counts[split] += 1
    _write_data_yaml(output_root, class_names or ["banana_plant"], include_test=counts["test"] > 0)
    summary = SplitSummary(
        images=len(items),
        groups=len(groups),
        train=counts["train"],
        val=counts["val"],
        test=counts["test"],
        missing_labels=missing_labels,
        output_dir=str(output_root),
    )
    (output_root / "split_summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    _write_group_assignments(output_root, assignments)
    return summary


def _normalize_ratios(train: float, val: float, test: float) -> dict[str, float]:
    if min(train, val, test) < 0:
        raise ValueError("Split ratios must be non-negative")
    total = train + val + test
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    return {"train": train / total, "val": val / total, "test": test / total}


def _read_group_manifest(manifest_csv: str | Path | None) -> dict[str, str]:
    if manifest_csv is None:
        return {}
    manifest_csv = Path(manifest_csv)
    lookup: dict[str, str] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image" not in (reader.fieldnames or []) or "group" not in (reader.fieldnames or []):
            raise ValueError("Manifest CSV must contain image and group columns")
        for row in reader:
            image = str(row["image"]).replace("\\", "/")
            group = str(row["group"])
            lookup[image] = group
            lookup[Path(image).name] = group
    return lookup


def _collect_items(image_dir: Path, label_dir: Path, group_lookup: dict[str, str]) -> list[SplitItem]:
    items: list[SplitItem] = []
    for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative = image.relative_to(image_dir)
        relative_key = relative.as_posix()
        group = group_lookup.get(relative_key) or group_lookup.get(image.name) or _default_group(relative)
        label = (label_dir / relative).with_suffix(".txt")
        items.append(SplitItem(image=image, label=label, relative=relative, group=group))
    if not items:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return items


def _default_group(relative: Path) -> str:
    parent = relative.parent.as_posix()
    return parent if parent and parent != "." else relative.stem


def _assign_groups(
    groups: dict[str, list[SplitItem]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[SplitItem]]:
    names = list(groups)
    random.Random(seed).shuffle(names)
    total_images = sum(len(items) for items in groups.values())
    targets = {split: total_images * ratio for split, ratio in ratios.items()}
    assignments: dict[str, list[SplitItem]] = {"train": [], "val": [], "test": []}
    for group_name in names:
        split = _split_with_most_remaining(assignments, targets)
        assignments[split].extend(groups[group_name])
    return assignments


def _split_with_most_remaining(
    assignments: dict[str, list[SplitItem]],
    targets: dict[str, float],
) -> str:
    remaining = {
        split: targets[split] - len(assignments[split])
        for split in ("train", "val", "test")
        if targets[split] > 0
    }
    return max(remaining, key=remaining.get)


def _copy_item(item: SplitItem, output_root: Path, split: str, write_empty_labels: bool) -> None:
    image_target = output_root / "images" / split / item.relative
    label_target = (output_root / "labels" / split / item.relative).with_suffix(".txt")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.image, image_target)
    if item.label.exists():
        shutil.copy2(item.label, label_target)
    elif write_empty_labels:
        label_target.write_text("", encoding="utf-8")


def _write_data_yaml(output_root: Path, class_names: list[str], include_test: bool) -> None:
    names = dict(enumerate(class_names))
    payload: dict[str, Any] = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if include_test:
        payload["test"] = "images/test"
    (output_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_group_assignments(output_root: Path, assignments: dict[str, list[SplitItem]]) -> None:
    rows = []
    for split, items in assignments.items():
        for item in items:
            rows.append({"split": split, "image": item.relative.as_posix(), "group": item.group})
    path = output_root / "split_assignments.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "group"])
        writer.writeheader()
        writer.writerows(rows)
