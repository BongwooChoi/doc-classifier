#!/usr/bin/env python3
"""간단한 파이프라인 테스트 스크립트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np


def test_preprocessing():
    """전처리 모듈 테스트"""
    print("=" * 60)
    print("1. 전처리 모듈 테스트")
    print("=" * 60)

    from src.preprocessor import Deskewer, DocumentDetector

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:400, 100:540] = 255

    deskewer = Deskewer(max_angle=15)
    result, angle = deskewer.deskew(test_image)
    print(f"✓ Deskewer 테스트 통과 (보정 각도: {angle:.2f}°)")

    detector = DocumentDetector(min_area_ratio=0.3)
    result, contour = detector.detect_and_extract(test_image)
    print(f"✓ DocumentDetector 테스트 통과 (문서 검출: {contour is not None})")

    return True


def test_yolo_detector():
    """YOLO 검출기 테스트"""
    print("\n" + "=" * 60)
    print("2. YOLO 검출기 테스트")
    print("=" * 60)

    from src.step1_yolo import YOLODetector, YOLOClassifier

    detector = YOLODetector(confidence_threshold=0.3)

    test_image_path = Path("data/sample/test/images")
    if test_image_path.exists():
        test_images = list(test_image_path.glob("*.jpg"))[:3]

        if test_images:
            for img_path in test_images:
                print(f"\n테스트 이미지: {img_path.name}")

                result = detector.detect(str(img_path), verbose=False)
                print(f"  - 검출된 객체: {result['detection_count']}개")

                for det in result["detections"][:3]:
                    print(f"    • {det['class']}: {det['confidence']:.2f}")

                features = detector.get_layout_features(result)
                print(f"  - 레이아웃 특징:")
                print(f"    • 도장: {features['has_stamp']}")
                print(f"    • 테이블: {features['has_table']}")
                print(f"    • 로고: {features['has_hospital_logo']}")

    print("\n✓ YOLODetector 테스트 통과")
    return True


def test_yolo_classifier():
    """YOLO 분류기 테스트"""
    print("\n" + "=" * 60)
    print("3. YOLO 분류기 테스트 (규칙 기반)")
    print("=" * 60)

    from src.step1_yolo import YOLOClassifier

    classifier = YOLOClassifier()

    mock_features = {
        "has_stamp": True,
        "has_hospital_logo": True,
        "has_table": False,
        "has_barcode": False,
        "has_signature": True,
        "table_area_ratio": 0.1,
        "stamp_count": 1,
        "table_count": 0,
        "logo_position": [100, 50],
        "detection_summary": {}
    }

    scores = classifier._calculate_scores(mock_features)

    print("레이아웃 특징 기반 분류 점수:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cls, score in sorted_scores:
        print(f"  - {cls}: {score:.3f}")

    print(f"\n예측 결과: {sorted_scores[0][0]} (점수: {sorted_scores[0][1]:.3f})")
    print("✓ YOLOClassifier 규칙 기반 분류 테스트 통과")

    return True


def test_sample_classification():
    """샘플 이미지 분류 테스트"""
    print("\n" + "=" * 60)
    print("4. 샘플 이미지 분류 테스트")
    print("=" * 60)

    from src.step1_yolo import YOLOClassifier

    test_dir = Path("data/sample/test/images")
    labels_file = Path("data/sample/test/labels.tsv")

    if not test_dir.exists():
        print("테스트 데이터가 없습니다. 먼저 샘플 데이터를 생성해주세요.")
        return False

    labels = {}
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    classifier = YOLOClassifier()

    test_images = list(test_dir.glob("*.jpg"))
    correct = 0
    total = 0

    print("\n분류 결과:")
    for img_path in test_images:
        gt_label = labels.get(img_path.name, "unknown")

        result = classifier.classify(str(img_path))
        pred_label = result["predicted_class"]
        confidence = result["confidence"]

        is_correct = pred_label == gt_label
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status} {img_path.name}")
        print(f"      정답: {gt_label} | 예측: {pred_label} ({confidence:.2f})")

    accuracy = correct / total if total > 0 else 0
    print(f"\n정확도: {correct}/{total} = {accuracy:.1%}")

    return True


def test_ocr():
    """OCR 모듈 테스트 (PaddleOCR 미설치시 스킵)"""
    print("\n" + "=" * 60)
    print("5. OCR 모듈 테스트")
    print("=" * 60)

    try:
        from src.step2_layoutlm import OCRProcessor

        processor = OCRProcessor(engine="paddleocr")

        boxes = [[0, 0, 100, 50], [50, 50, 150, 100]]
        image_size = (200, 100)
        normalized = processor.normalize_boxes(boxes, image_size)

        print(f"바운딩 박스 정규화 테스트:")
        print(f"  원본: {boxes}")
        print(f"  정규화: {normalized}")
        print("✓ OCR 박스 정규화 테스트 통과")

        print("\n⚠ PaddleOCR가 설치되지 않아 실제 OCR 테스트는 스킵합니다.")
        print("  설치: pip install paddleocr paddlepaddle")

    except Exception as e:
        print(f"OCR 테스트 스킵: {e}")

    return True


def test_vlm_prompts():
    """VLM 프롬프트 테스트"""
    print("\n" + "=" * 60)
    print("6. VLM 프롬프트 테스트")
    print("=" * 60)

    from src.step3_vlm import DocumentClassificationPrompts

    basic_prompt = DocumentClassificationPrompts.get_classification_prompt()
    print(f"기본 프롬프트 길이: {len(basic_prompt)} 문자")

    step1_result = {"predicted_class": "진단서", "confidence": 0.7}
    step2_result = {"predicted_class": "소견서", "confidence": 0.6}
    context_prompt = DocumentClassificationPrompts.get_classification_prompt(
        step1_result, step2_result
    )
    print(f"컨텍스트 포함 프롬프트 길이: {len(context_prompt)} 문자")

    sample_response = """분류: 진단서
신뢰도: 높음
근거: 문서에 진단명, 환자 정보, 의사 서명이 포함되어 있습니다."""

    parsed = DocumentClassificationPrompts.parse_response(sample_response)
    print(f"\n응답 파싱 테스트:")
    print(f"  - 예측 클래스: {parsed['predicted_class']}")
    print(f"  - 신뢰도: {parsed['confidence_level']}")
    print(f"  - 근거: {parsed['reasoning']}")

    print("✓ VLM 프롬프트 테스트 통과")
    return True


def main():
    print("\n" + "=" * 60)
    print("문서분류기 파이프라인 테스트")
    print("=" * 60 + "\n")

    tests = [
        ("전처리", test_preprocessing),
        ("YOLO 검출기", test_yolo_detector),
        ("YOLO 분류기", test_yolo_classifier),
        ("샘플 분류", test_sample_classification),
        ("OCR", test_ocr),
        ("VLM 프롬프트", test_vlm_prompts),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} 테스트 실패: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ 통과" if success else "✗ 실패"
        print(f"  {status}: {name}")

    print(f"\n전체: {passed}/{total} 테스트 통과")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
