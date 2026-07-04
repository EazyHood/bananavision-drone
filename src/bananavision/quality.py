from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .pipeline import IMAGE_EXTENSIONS


def audit_dataset_quality(
    data_yaml: str | Path,
    require_val: bool = True,
    require_test: bool = False,
) -> dict[str, Any]:
    data_yaml = Path(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(config.get("path", data_yaml.parent))
    names = _names(config.get("names", {0: "banana_plant"}))
    split_paths = _split_paths(root, config)
    issues: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    image_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()

    if require_val and "val" not in split_paths:
        issues.append(_issue("missing_split", "fail", "Validation split is missing", {"split": "val"}))
    if require_test and "test" not in split_paths:
        issues.append(_issue("missing_split", "fail", "Test split is missing", {"split": "test"}))

    for split, image_dir in split_paths.items():
        report = _audit_split(split, image_dir, names, issues, image_hashes, class_counts)
        split_reports[split] = report

    issues.extend(_duplicate_hash_issues(image_hashes))
    issues.extend(_group_leakage_issues(root / "split_assignments.csv"))
    status = _status(issues)
    return {
        "data_yaml": str(data_yaml),
        "root": str(root),
        "status": status,
        "class_names": names,
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "splits": split_reports,
        "issues": issues,
    }


def write_quality_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _split_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        if split in config:
            paths[split] = root / str(config[split])
    return paths


def _audit_split(
    split: str,
    image_dir: Path,
    names: dict[int, str],
    issues: list[dict[str, Any]],
    image_hashes: dict[str, list[tuple[str, str]]],
    class_counts: Counter[int],
) -> dict[str, Any]:
    if not image_dir.exists():
        issues.append(_issue("missing_image_dir", "fail", f"Image directory missing: {image_dir}", {"split": split}))
        return {"images": 0, "labels": 0, "objects": 0, "empty_labels": 0}
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = 0
    objects = 0
    empty_labels = 0
    for image in images:
        digest = _file_sha1(image)
        image_hashes[digest].append((split, str(image)))
        label = _label_for_image(image)
        if not label.exists():
            issues.append(_issue("missing_label", "fail", f"Missing label for {image}", {"split": split, "image": str(image)}))
            continue
        labels += 1
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line_number, line in enumerate(lines, start=1):
            parsed = _parse_label_line(line, names, split, label, line_number)
            if parsed["issues"]:
                issues.extend(parsed["issues"])
            else:
                class_counts[parsed["class_id"]] += 1
                objects += 1
    if not images:
        issues.append(_issue("empty_split", "warn", f"No images found in split {split}", {"split": split}))
    return {"images": len(images), "labels": labels, "objects": objects, "empty_labels": empty_labels}


def _parse_label_line(
    line: str,
    names: dict[int, str],
    split: str,
    label: Path,
    line_number: int,
) -> dict[str, Any]:
    details = {"split": split, "label": str(label), "line": line_number}
    issues: list[dict[str, Any]] = []
    parts = line.split()
    if len(parts) < 5:
        issues.append(_issue("invalid_label", "fail", "Label line has too few values", details))
        return {"issues": issues, "class_id": -1}
    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        issues.append(_issue("invalid_label", "fail", "Label line contains non-numeric values", details))
        return {"issues": issues, "class_id": -1}
    if class_id not in names:
        issues.append(_issue("unknown_class", "fail", f"Unknown class id {class_id}", {**details, "class_id": class_id}))
    if any(value < 0.0 or value > 1.0 for value in values):
        issues.append(_issue("coordinate_range", "fail", "Coordinates must be normalized between 0 and 1", details))
    if len(values) == 4:
        if values[2] <= 0 or values[3] <= 0:
            issues.append(_issue("invalid_box", "fail", "YOLO box width/height must be positive", details))
    elif len(values) < 6 or len(values) % 2 != 0:
        issues.append(_issue("invalid_polygon", "fail", "Segmentation labels need x/y pairs for at least 3 points", details))
    return {"issues": issues, "class_id": class_id}


def _duplicate_hash_issues(image_hashes: dict[str, list[tuple[str, str]]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for digest, locations in image_hashes.items():
        splits = {split for split, _ in locations}
        if len(splits) > 1:
            issues.append(
                _issue(
                    "duplicate_image_across_splits",
                    "fail",
                    "Identical image bytes appear in multiple splits",
                    {"sha1": digest, "locations": [{"split": split, "image": image} for split, image in locations]},
                )
            )
    return issues


def _group_leakage_issues(assignments_csv: Path) -> list[dict[str, Any]]:
    if not assignments_csv.exists():
        return []
    groups: dict[str, set[str]] = defaultdict(set)
    with assignments_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "split" not in (reader.fieldnames or []) or "group" not in (reader.fieldnames or []):
            return [_issue("invalid_assignments", "warn", "split_assignments.csv lacks split/group columns", {})]
        for row in reader:
            groups[str(row["group"])].add(str(row["split"]))
    return [
        _issue("group_leakage", "fail", "Group appears in more than one split", {"group": group, "splits": sorted(splits)})
        for group, splits in groups.items()
        if len(splits) > 1
    ]


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _names(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    return {0: "banana_plant"}


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(code: str, severity: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def _status(issues: list[dict[str, Any]]) -> str:
    if any(issue["severity"] == "fail" for issue in issues):
        return "fail"
    if any(issue["severity"] == "warn" for issue in issues):
        return "warn"
    return "pass"
