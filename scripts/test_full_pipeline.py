#!/usr/bin/env python3
"""전체 파이프라인 통합 테스트"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DocumentClassificationPipeline


def test_pipeline_step1_only():
    """Step 1만 사용하는 파이프라인 테스트"""
    print("\n" + "=" * 60)
    print("Step 1 (YOLO) 전용 파이프라인 테스트")
    print("=" * 60)

    pipeline = DocumentClassificationPipeline(
        enable_step1=True,
        enable_step2=False,  # LayoutLM 비활성화
        enable_step3=False,  # VLM 비활성화
        enable_preprocessing=True
    )

    test_dir = Path("data/sample/test/images")
    labels_file = Path("data/sample/test/labels.tsv")

    labels = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]

    test_images = list(test_dir.glob("*.jpg"))[:6]

    results = []
    for img_path in test_images:
        result = pipeline.classify(str(img_path))
        gt_label = labels.get(img_path.name, "unknown")

        results.append({
            "file": img_path.name,
            "predicted": result.predicted_class,
            "ground_truth": gt_label,
            "confidence": result.confidence,
            "final_step": result.final_step,
            "correct": result.predicted_class == gt_label
        })

        print(f"\n{img_path.name}:")
        print(f"  정답: {gt_label}")
        print(f"  예측: {result.predicted_class} (신뢰도: {result.confidence:.2%})")
        print(f"  최종 단계: Step {result.final_step}")
        print(f"  처리 시간: {result.processing_time:.3f}초")

    correct = sum(1 for r in results if r["correct"])
    print(f"\n정확도: {correct}/{len(results)} = {correct/len(results):.1%}")

    return results


def test_mock_step2():
    """Step 2 (LayoutLM) 모킹 테스트 - 모델 다운로드 없이"""
    print("\n" + "=" * 60)
    print("Step 2 (LayoutLM) 모킹 테스트")
    print("=" * 60)

    from src.step2_layoutlm import LayoutLMClassifier

    classifier = LayoutLMClassifier(
        confidence_threshold=0.7
    )

    print("LayoutLMClassifier 초기화 성공")
    print(f"  - 모델: {classifier.model_name}")
    print(f"  - 신뢰도 임계값: {classifier.confidence_threshold}")
    print(f"  - 문서 클래스: {classifier.DOCUMENT_CLASSES}")

    mock_result = {
        "all_probabilities": {
            "진단서": 0.85,
            "소견서": 0.08,
            "보험금청구서": 0.03,
            "입퇴원확인서": 0.02,
            "의료비영수증": 0.01,
            "처방전": 0.01
        }
    }

    top_3 = classifier.get_top_predictions(mock_result, top_k=3)
    print(f"\nTop-3 예측:")
    for cls, prob in top_3:
        print(f"  - {cls}: {prob:.2%}")


def test_statistics():
    """통계 기능 테스트"""
    print("\n" + "=" * 60)
    print("파이프라인 통계 기능 테스트")
    print("=" * 60)

    from src.pipeline import ClassificationResult

    mock_results = [
        ClassificationResult("진단서", 0.9, 1, processing_time=0.5),
        ClassificationResult("진단서", 0.85, 2, processing_time=1.2),
        ClassificationResult("소견서", 0.95, 1, processing_time=0.4),
        ClassificationResult("처방전", 0.7, 3, processing_time=2.5),
        ClassificationResult("의료비영수증", 0.88, 2, processing_time=1.1),
        ClassificationResult("보험금청구서", 0.92, 1, processing_time=0.6),
    ]

    pipeline = DocumentClassificationPipeline(
        enable_step1=False,
        enable_step2=False,
        enable_step3=False
    )

    stats = pipeline.get_statistics(mock_results)

    print(f"\n총 문서 수: {stats['total_documents']}")
    print(f"\n클래스별 분포:")
    for cls, count in stats["class_distribution"].items():
        if count > 0:
            print(f"  - {cls}: {count}개")

    print(f"\n단계별 분류:")
    for step, count in stats["step_distribution"].items():
        print(f"  - {step}: {count}개")

    print(f"\nVLM 사용률: {stats['vlm_usage_rate']:.1%}")
    print(f"평균 처리 시간: {stats['average_processing_time']:.2f}초")
    print(f"평균 신뢰도: {stats['average_confidence']:.2%}")


def test_batch_classification():
    """배치 분류 테스트"""
    print("\n" + "=" * 60)
    print("배치 분류 테스트")
    print("=" * 60)

    pipeline = DocumentClassificationPipeline(
        enable_step1=True,
        enable_step2=False,
        enable_step3=False,
        enable_preprocessing=False
    )

    test_dir = Path("data/sample/test/images")
    test_images = [str(p) for p in list(test_dir.glob("*.jpg"))[:4]]

    print(f"배치 처리할 이미지: {len(test_images)}개")

    results = pipeline.batch_classify(test_images, show_progress=False)

    print(f"\n배치 분류 결과:")
    class_counts = Counter(r.predicted_class for r in results)
    for cls, count in class_counts.most_common():
        print(f"  - {cls}: {count}개")

    total_time = sum(r.processing_time or 0 for r in results)
    print(f"\n총 처리 시간: {total_time:.2f}초")
    print(f"이미지당 평균: {total_time/len(results):.2f}초")


def main():
    print("=" * 60)
    print("문서분류기 전체 파이프라인 테스트")
    print("=" * 60)

    test_statistics()

    test_mock_step2()

    test_pipeline_step1_only()

    test_batch_classification()

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)

    print("\n⚠️ 참고사항:")
    print("  - 현재 YOLO 모델은 문서 레이아웃 검출용으로 학습되지 않았습니다.")
    print("  - 실제 사용을 위해서는 다음 단계가 필요합니다:")
    print("    1. 문서 레이아웃 데이터로 YOLO 모델 학습")
    print("    2. 문서 분류 데이터로 LayoutLM 모델 학습")
    print("    3. VLM API 키 설정 (OpenAI 또는 Anthropic)")


if __name__ == "__main__":
    main()
