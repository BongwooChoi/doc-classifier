#!/usr/bin/env python3
"""데이터 준비 스크립트"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import random

import yaml


def create_yolo_data_yaml(
    data_dir: Path,
    output_path: Path,
    classes: List[str]
) -> None:
    """YOLO 데이터셋 YAML 생성"""
    config = {
        "path": str(data_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"YOLO 데이터 YAML 생성: {output_path}")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, int]:
    """데이터셋 분할"""
    random.seed(seed)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []

    for ext in image_extensions:
        images.extend(source_dir.glob(f"*{ext}"))
        images.extend(source_dir.glob(f"*{ext.upper()}"))

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():
        split_image_dir = output_dir / "images" / split_name
        split_label_dir = output_dir / "labels" / split_name

        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, split_image_dir / image_path.name)

            label_path = image_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, split_label_dir / label_path.name)

    return {
        "train": len(train_images),
        "val": len(val_images),
        "test": len(test_images)
    }


def create_layoutlm_annotations(
    image_dir: Path,
    ocr_dir: Path,
    labels_file: Path,
    output_dir: Path
) -> int:
    """LayoutLM용 어노테이션 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    count = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for ext in image_extensions:
        for image_path in image_dir.glob(f"*{ext}"):
            image_name = image_path.stem

            ocr_path = ocr_dir / f"{image_name}.json"
            if not ocr_path.exists():
                continue

            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            annotation = {
                "image": image_path.name,
                "words": ocr_data.get("words", []),
                "boxes": ocr_data.get("boxes", []),
                "label": labels.get(image_name, labels.get(image_path.name))
            }

            output_path = output_dir / f"{image_name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            count += 1

    return count


def verify_dataset_structure(data_dir: Path, dataset_type: str) -> bool:
    """데이터셋 구조 검증"""
    if dataset_type == "yolo":
        required_dirs = [
            "images/train",
            "images/val",
            "labels/train",
            "labels/val"
        ]
    elif dataset_type == "layoutlm":
        required_dirs = [
            "images",
            "annotations"
        ]
    else:
        print(f"알 수 없는 데이터셋 타입: {dataset_type}")
        return False

    missing = []
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if not full_path.exists():
            missing.append(dir_path)

    if missing:
        print(f"누락된 디렉토리: {missing}")
        return False

    print(f"데이터셋 구조 검증 완료: {data_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="데이터 준비 스크립트")

    subparsers = parser.add_subparsers(dest="command", help="명령")

    split_parser = subparsers.add_parser("split", help="데이터셋 분할")
    split_parser.add_argument("source", help="원본 데이터 디렉토리")
    split_parser.add_argument("output", help="출력 디렉토리")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)

    yolo_parser = subparsers.add_parser("yolo-yaml", help="YOLO 데이터 YAML 생성")
    yolo_parser.add_argument("data_dir", help="데이터 디렉토리")
    yolo_parser.add_argument("--output", default="data/annotations/yolo_data.yaml")
    yolo_parser.add_argument(
        "--classes",
        nargs="+",
        default=["document", "stamp", "hospital_logo", "barcode", "table", "signature"]
    )

    layoutlm_parser = subparsers.add_parser("layoutlm-anno", help="LayoutLM 어노테이션 생성")
    layoutlm_parser.add_argument("image_dir", help="이미지 디렉토리")
    layoutlm_parser.add_argument("ocr_dir", help="OCR 결과 디렉토리")
    layoutlm_parser.add_argument("labels_file", help="레이블 파일 (TSV)")
    layoutlm_parser.add_argument("--output", default="data/annotations/layoutlm")

    verify_parser = subparsers.add_parser("verify", help="데이터셋 구조 검증")
    verify_parser.add_argument("data_dir", help="데이터 디렉토리")
    verify_parser.add_argument("--type", choices=["yolo", "layoutlm"], required=True)

    args = parser.parse_args()

    if args.command == "split":
        counts = split_dataset(
            Path(args.source),
            Path(args.output),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
        print(f"데이터셋 분할 완료: {counts}")

    elif args.command == "yolo-yaml":
        create_yolo_data_yaml(
            Path(args.data_dir),
            Path(args.output),
            args.classes
        )

    elif args.command == "layoutlm-anno":
        count = create_layoutlm_annotations(
            Path(args.image_dir),
            Path(args.ocr_dir),
            Path(args.labels_file),
            Path(args.output)
        )
        print(f"LayoutLM 어노테이션 생성: {count}개")

    elif args.command == "verify":
        success = verify_dataset_structure(Path(args.data_dir), args.type)
        exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
