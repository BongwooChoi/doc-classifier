#!/usr/bin/env python3
"""API 테스트 스크립트

사용법:
1. 먼저 API 서버를 시작합니다: python run_api.py
2. 다른 터미널에서 이 스크립트를 실행합니다: python scripts/test_api.py
"""

import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests 패키지가 필요합니다: pip install requests")
    sys.exit(1)

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

API_BASE_URL = "http://localhost:8000"


def test_health():
    """헬스 체크 테스트"""
    print("\n[1] 헬스 체크")
    print("-" * 40)
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  상태: {data['status']}")
            print(f"  YOLO 로드: {data['models_loaded']['yolo']}")
            print(f"  LayoutLM 로드: {data['models_loaded']['layoutlm']}")
            return True
        else:
            print(f"  오류: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  연결 실패: API 서버가 실행 중인지 확인하세요.")
        print("  실행 방법: python run_api.py")
        return False


def test_classes():
    """클래스 목록 테스트"""
    print("\n[2] 지원 문서 클래스")
    print("-" * 40)
    response = requests.get(f"{API_BASE_URL}/classes")
    if response.status_code == 200:
        classes = response.json()
        for i, cls in enumerate(classes, 1):
            print(f"  {i}. {cls}")
        return True
    return False


def test_single_classification():
    """단일 이미지 분류 테스트"""
    print("\n[3] 단일 이미지 분류 테스트")
    print("-" * 40)

    # 테스트 이미지 선택 (각 유형별 1개씩)
    test_files = {
        "진단서": list(VAL_IMAGES_PATH.glob("diagnosis*.jpg"))[:1],
        "소견서": list(VAL_IMAGES_PATH.glob("opinion*.jpg"))[:1],
        "보험금청구서": list(VAL_IMAGES_PATH.glob("insurance*.jpg"))[:1] or list(VAL_IMAGES_PATH.glob("claim*.jpg"))[:1],
        "입퇴원확인서": list(VAL_IMAGES_PATH.glob("admission*.jpg"))[:1],
        "의료비영수증": list(VAL_IMAGES_PATH.glob("receipt*.jpg"))[:1],
        "처방전": list(VAL_IMAGES_PATH.glob("prescription*.jpg"))[:1],
    }

    correct = 0
    total = 0

    for expected_class, files in test_files.items():
        if not files:
            print(f"  {expected_class}: 테스트 이미지 없음")
            continue

        file_path = files[0]
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{API_BASE_URL}/classify",
                files={"file": (file_path.name, f, "image/jpeg")},
                params={"use_mock_ocr": True, "force_step2": True}
            )

        if response.status_code == 200:
            result = response.json()
            pred_class = result["predicted_class"]
            confidence = result["confidence"]
            final_step = result["final_step"]

            is_correct = pred_class == expected_class
            status = "O" if is_correct else "X"
            correct += 1 if is_correct else 0
            total += 1

            print(f"  {status} {expected_class}: 예측={pred_class} "
                  f"(conf: {confidence:.2f}, step: {final_step})")
        else:
            print(f"  {expected_class}: HTTP {response.status_code}")

    if total > 0:
        print(f"\n  정확도: {correct}/{total} ({correct/total*100:.1f}%)")


def test_batch_classification():
    """배치 분류 테스트"""
    print("\n[4] 배치 분류 테스트")
    print("-" * 40)

    # 3개 이미지로 테스트
    test_images = list(VAL_IMAGES_PATH.glob("*.jpg"))[:3]

    if not test_images:
        print("  테스트 이미지 없음")
        return

    files = []
    for img_path in test_images:
        files.append(("files", (img_path.name, open(img_path, "rb"), "image/jpeg")))

    response = requests.post(
        f"{API_BASE_URL}/classify/batch",
        files=files,
        params={"use_mock_ocr": True}
    )

    # 파일 핸들 닫기
    for _, (_, f, _) in files:
        f.close()

    if response.status_code == 200:
        data = response.json()
        print(f"  처리된 파일: {data['total']}개")
        for item in data["results"]:
            if item["success"]:
                result = item["result"]
                print(f"  - {item['filename']}: {result['predicted_class']} "
                      f"({result['confidence']:.2f})")
            else:
                print(f"  - {item['filename']}: 오류 - {item['error']}")


def test_detailed_result():
    """상세 결과 테스트"""
    print("\n[5] 상세 분류 결과")
    print("-" * 40)

    # 소견서 테스트 (Step 1에서는 진단서로 분류될 수 있음)
    test_file = list(VAL_IMAGES_PATH.glob("opinion*.jpg"))
    if not test_file:
        print("  테스트 이미지 없음")
        return

    file_path = test_file[0]
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            files={"file": (file_path.name, f, "image/jpeg")},
            params={"use_mock_ocr": True, "force_step2": True}
        )

    if response.status_code == 200:
        result = response.json()
        print(f"  파일: {file_path.name}")
        print(f"  최종 분류: {result['predicted_class']}")
        print(f"  신뢰도: {result['confidence']:.4f}")
        print(f"  최종 단계: Step {result['final_step']}")
        print(f"  처리 시간: {result['processing_time']:.3f}초")

        if result.get("step1_result"):
            print("\n  [Step 1 YOLO 결과]")
            s1 = result["step1_result"]
            print(f"    예측: {s1['predicted_class']} ({s1['confidence']:.2f})")
            print(f"    Step 2 필요: {s1['requires_step2']}")

        if result.get("step2_result"):
            print("\n  [Step 2 LayoutLM 결과]")
            s2 = result["step2_result"]
            print(f"    예측: {s2['predicted_class']} ({s2['confidence']:.4f})")
            print(f"    OCR 단어 수: {s2['ocr_words_count']}")
            print(f"    Mock OCR 사용: {s2['mock_ocr_used']}")
            print("\n    클래스별 확률:")
            for cls, prob in sorted(s2["all_probabilities"].items(), key=lambda x: -x[1])[:3]:
                print(f"      {cls}: {prob:.4f}")


def main():
    print("=" * 60)
    print("문서 분류 API 테스트")
    print("=" * 60)

    # 헬스 체크 (서버 연결 확인)
    if not test_health():
        return

    # 클래스 목록
    test_classes()

    # 단일 분류 테스트
    test_single_classification()

    # 배치 분류 테스트
    test_batch_classification()

    # 상세 결과 테스트
    test_detailed_result()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
