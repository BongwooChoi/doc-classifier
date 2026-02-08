#!/usr/bin/env python3
"""Step 1 + Step 2 통합 파이프라인 테스트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from collections import defaultdict

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

# 파일명 매핑
FILENAME_TO_CLASS = {
    "diagnosis": "진단서",
    "opinion": "소견서",
    "claim": "보험금청구서",
    "insurance": "보험금청구서",
    "admission": "입퇴원확인서",
    "receipt": "의료비영수증",
    "prescription": "처방전"
}


def get_class_from_filename(filename: str) -> str:
    """파일명에서 클래스 추출"""
    for prefix, cls in FILENAME_TO_CLASS.items():
        if filename.startswith(prefix):
            return cls
    return "unknown"


def main():
    print("=" * 70)
    print("3단계 파이프라인 통합 테스트 (Step 1 + Step 2)")
    print("=" * 70)

    # 파이프라인 임포트 및 초기화
    from src.pipeline import DocumentClassificationPipeline

    print("\n파이프라인 초기화 중...")
    pipeline = DocumentClassificationPipeline(
        enable_preprocessing=False,  # 전처리 비활성화 (테스트용)
        enable_step1=True,
        enable_step2=True,
        enable_step3=False  # VLM은 API 키 필요하므로 비활성화
    )
    print("파이프라인 초기화 완료!")

    # 검증 이미지 목록
    val_images = list(VAL_IMAGES_PATH.glob("*.png")) + list(VAL_IMAGES_PATH.glob("*.jpg"))
    print(f"\n검증 이미지 수: {len(val_images)}")

    # 결과 집계
    results_by_type = defaultdict(lambda: {"correct": 0, "incorrect": 0, "predictions": []})
    step_usage = {1: 0, 2: 0}

    print("\n" + "-" * 70)
    print("분류 테스트 진행 중...")
    print("-" * 70)

    for img_path in sorted(val_images):
        true_label = get_class_from_filename(img_path.stem)
        if true_label == "unknown":
            continue

        # 파이프라인 분류
        result = pipeline.classify(str(img_path))
        pred_label = result.predicted_class
        confidence = result.confidence
        final_step = result.final_step

        # 결과 기록
        step_usage[final_step] += 1
        is_correct = (pred_label == true_label)

        results_by_type[true_label]["predictions"].append({
            "file": img_path.name,
            "predicted": pred_label,
            "confidence": confidence,
            "final_step": final_step,
            "correct": is_correct
        })

        if is_correct:
            results_by_type[true_label]["correct"] += 1
        else:
            results_by_type[true_label]["incorrect"] += 1

    # 오분류 출력
    print("\n오분류된 샘플:")
    any_error = False
    doc_types = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]
    for doc_type in doc_types:
        for pred in results_by_type[doc_type]["predictions"]:
            if not pred["correct"]:
                any_error = True
                print(f"  {pred['file']}: 정답={doc_type}, 예측={pred['predicted']} "
                      f"(conf: {pred['confidence']:.2f}, step: {pred['final_step']})")

    if not any_error:
        print("  (오분류 없음!)")

    # 문서 유형별 정확도
    print("\n" + "=" * 70)
    print("문서 유형별 정확도")
    print("=" * 70)

    total_correct = 0
    total_count = 0

    for doc_type in doc_types:
        stats = results_by_type[doc_type]
        correct = stats["correct"]
        total = correct + stats["incorrect"]
        total_correct += correct
        total_count += total
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"  {doc_type}: {correct}/{total} ({accuracy:.1f}%)")

    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    print(f"\n전체 정확도: {total_correct}/{total_count} ({overall_accuracy:.1f}%)")

    # Step별 사용량
    print("\n" + "=" * 70)
    print("Step별 분류 비율")
    print("=" * 70)
    for step, count in step_usage.items():
        ratio = (count / total_count * 100) if total_count > 0 else 0
        print(f"  Step {step}: {count}개 ({ratio:.1f}%)")

    # 샘플 출력
    print("\n" + "=" * 70)
    print("샘플 분류 결과 (각 유형별 첫 2개)")
    print("=" * 70)

    for doc_type in doc_types:
        print(f"\n[{doc_type}]")
        predictions = results_by_type[doc_type]["predictions"][:2]
        for pred in predictions:
            status = "O" if pred["correct"] else "X"
            print(f"  {status} {pred['file']}: {pred['predicted']} "
                  f"(conf: {pred['confidence']:.2f}, step: {pred['final_step']})")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
