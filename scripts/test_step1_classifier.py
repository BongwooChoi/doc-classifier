#!/usr/bin/env python3
"""Step 1 YOLO 분류기 테스트 스크립트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from collections import defaultdict

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "yolo_document_layout" / "weights" / "best.pt"
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

# 파일명에서 문서 유형 추출
def get_document_type_from_filename(filename: str) -> str:
    """파일명에서 문서 유형 추출"""
    name_map = {
        "diagnosis": "진단서",
        "opinion": "소견서",
        "claim": "보험금청구서",
        "insurance": "보험금청구서",  # insurance도 보험금청구서로 매핑
        "admission": "입퇴원확인서",
        "receipt": "의료비영수증",
        "prescription": "처방전"
    }
    for key, value in name_map.items():
        if filename.startswith(key):
            return value
    return "unknown"

def main():
    from src.step1_yolo.detector import YOLODetector
    from src.step1_yolo.classifier import YOLOClassifier

    print("=" * 70)
    print("Step 1: YOLO 기반 문서 분류기 테스트")
    print("=" * 70)

    # 모델 로드
    print(f"\n모델 경로: {MODEL_PATH}")
    detector = YOLODetector(model_path=str(MODEL_PATH), confidence_threshold=0.5)
    classifier = YOLOClassifier(detector=detector, confidence_threshold=0.7)

    # 검증 이미지 목록
    val_images = list(VAL_IMAGES_PATH.glob("*.png")) + list(VAL_IMAGES_PATH.glob("*.jpg"))
    print(f"검증 이미지 수: {len(val_images)}")

    # 문서 유형별 결과 집계
    results_by_type = defaultdict(lambda: {"correct": 0, "incorrect": 0, "predictions": []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    print("\n" + "-" * 70)
    print("개별 분류 결과")
    print("-" * 70)

    for img_path in sorted(val_images):
        true_label = get_document_type_from_filename(img_path.stem)

        # 분류 실행
        result = classifier.classify(str(img_path))
        predicted = result["predicted_class"]
        confidence = result["confidence"]

        # 결과 기록
        is_correct = (predicted == true_label)
        results_by_type[true_label]["predictions"].append({
            "file": img_path.name,
            "predicted": predicted,
            "confidence": confidence,
            "correct": is_correct
        })

        if is_correct:
            results_by_type[true_label]["correct"] += 1
        else:
            results_by_type[true_label]["incorrect"] += 1

        confusion_matrix[true_label][predicted] += 1

        # 결과 출력 (오분류만)
        if not is_correct:
            print(f"  {img_path.name}: 정답={true_label}, 예측={predicted} ({confidence:.2f}) ❌")

    # 문서 유형별 정확도
    print("\n" + "=" * 70)
    print("문서 유형별 정확도")
    print("=" * 70)

    total_correct = 0
    total_count = 0

    for doc_type in ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]:
        stats = results_by_type[doc_type]
        correct = stats["correct"]
        total = correct + stats["incorrect"]
        total_correct += correct
        total_count += total
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"  {doc_type}: {correct}/{total} ({accuracy:.1f}%)")

    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    print(f"\n전체 정확도: {total_correct}/{total_count} ({overall_accuracy:.1f}%)")

    # 혼동 행렬
    print("\n" + "=" * 70)
    print("혼동 행렬 (행: 실제, 열: 예측)")
    print("=" * 70)

    doc_types = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]

    # 헤더
    header = "실제\\예측   "
    for dt in doc_types:
        header += f"{dt[:3]:>5}"
    print(header)
    print("-" * 50)

    # 각 행
    for true_type in doc_types:
        row = f"{true_type[:5]:>8}    "
        for pred_type in doc_types:
            count = confusion_matrix[true_type][pred_type]
            row += f"{count:>5}"
        print(row)

    # 상위 예측 샘플
    print("\n" + "=" * 70)
    print("상세 분류 결과 샘플 (각 유형별 첫 2개)")
    print("=" * 70)

    for doc_type in doc_types:
        print(f"\n[{doc_type}]")
        predictions = results_by_type[doc_type]["predictions"][:2]
        for pred in predictions:
            status = "✓" if pred["correct"] else "✗"
            print(f"  {status} {pred['file']}: {pred['predicted']} ({pred['confidence']:.2f})")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
