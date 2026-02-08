#!/usr/bin/env python3
"""파이프라인 평가 스크립트"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DocumentClassificationPipeline
from src.utils.logger import get_logger


def load_ground_truth(labels_file: Path) -> Dict[str, str]:
    """정답 레이블 로드

    Args:
        labels_file: 레이블 파일 (TSV 또는 JSON)

    Returns:
        {파일명: 레이블} 딕셔너리
    """
    labels = {}

    if labels_file.suffix == ".json":
        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    labels[item["file"]] = item["label"]
            else:
                labels = data
    else:
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    return labels


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    classes: List[str]
) -> Dict:
    """평가 메트릭 계산

    Args:
        predictions: 예측 레이블 리스트
        ground_truths: 정답 레이블 리스트
        classes: 클래스 목록

    Returns:
        메트릭 딕셔너리
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            correct += 1
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    accuracy = correct / total if total > 0 else 0

    class_metrics = {}
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[cls] + fn[cls]
        }

    precisions = [m["precision"] for m in class_metrics.values() if m["support"] > 0]
    recalls = [m["recall"] for m in class_metrics.values() if m["support"] > 0]
    f1s = [m["f1"] for m in class_metrics.values() if m["support"] > 0]

    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion": {
            "true_positives": dict(tp),
            "false_positives": dict(fp),
            "false_negatives": dict(fn)
        }
    }


def print_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    classes: List[str]
) -> None:
    """혼동 행렬 출력"""
    matrix = defaultdict(lambda: defaultdict(int))

    for pred, gt in zip(predictions, ground_truths):
        matrix[gt][pred] += 1

    max_len = max(len(cls) for cls in classes)

    print("\n혼동 행렬:")
    print(" " * (max_len + 2), end="")
    for cls in classes:
        print(f"{cls[:6]:>8}", end="")
    print()

    for gt_cls in classes:
        print(f"{gt_cls:<{max_len + 2}}", end="")
        for pred_cls in classes:
            count = matrix[gt_cls][pred_cls]
            print(f"{count:>8}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="문서 분류 파이프라인 평가")

    parser.add_argument(
        "--test-dir",
        required=True,
        help="테스트 이미지 디렉토리"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="정답 레이블 파일 (TSV 또는 JSON)"
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="설정 파일 경로"
    )

    parser.add_argument(
        "--output",
        help="결과 저장 파일 (JSON)"
    )

    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Step 1 (YOLO) 건너뛰기"
    )

    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Step 2 (LayoutLM) 건너뛰기"
    )

    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Step 3 (VLM) 건너뛰기"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력"
    )

    args = parser.parse_args()
    logger = get_logger("evaluate", level="DEBUG" if args.verbose else "INFO")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"테스트 디렉토리를 찾을 수 없습니다: {test_dir}")
        return 1

    labels_file = Path(args.labels)
    if not labels_file.exists():
        logger.error(f"레이블 파일을 찾을 수 없습니다: {labels_file}")
        return 1

    ground_truth = load_ground_truth(labels_file)
    logger.info(f"정답 레이블 로드: {len(ground_truth)}개")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_dir.glob(f"*{ext}"))
        test_images.extend(test_dir.glob(f"*{ext.upper()}"))

    test_images = [img for img in test_images if img.name in ground_truth]
    logger.info(f"평가할 이미지: {len(test_images)}개")

    if not test_images:
        logger.error("평가할 이미지가 없습니다")
        return 1

    pipeline = DocumentClassificationPipeline(
        config_path=args.config,
        enable_step1=not args.skip_step1,
        enable_step2=not args.skip_step2,
        enable_step3=not args.skip_step3
    )

    predictions = []
    ground_truths = []
    results = []
    step_counts = {1: 0, 2: 0, 3: 0}

    for image_path in test_images:
        logger.debug(f"처리 중: {image_path.name}")

        result = pipeline.classify(str(image_path))

        predictions.append(result.predicted_class)
        ground_truths.append(ground_truth[image_path.name])
        step_counts[result.final_step] += 1

        results.append({
            "file": image_path.name,
            "predicted": result.predicted_class,
            "ground_truth": ground_truth[image_path.name],
            "correct": result.predicted_class == ground_truth[image_path.name],
            "confidence": result.confidence,
            "final_step": result.final_step
        })

    metrics = calculate_metrics(
        predictions,
        ground_truths,
        pipeline.DOCUMENT_CLASSES
    )

    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"\n전체 정확도: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    print("\n클래스별 성능:")
    print(f"{'클래스':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for cls, m in metrics["class_metrics"].items():
        print(f"{cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")

    print(f"\n단계별 분류 비율:")
    total = sum(step_counts.values())
    for step, count in step_counts.items():
        ratio = count / total * 100 if total > 0 else 0
        print(f"  Step {step}: {count}개 ({ratio:.1f}%)")

    print_confusion_matrix(predictions, ground_truths, pipeline.DOCUMENT_CLASSES)

    if args.output:
        output_data = {
            "metrics": metrics,
            "step_distribution": step_counts,
            "results": results
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
