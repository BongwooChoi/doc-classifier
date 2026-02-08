#!/usr/bin/env python3
"""학습된 YOLO 모델 테스트 스크립트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from pathlib import Path
import random

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "yolo_document_layout" / "weights" / "best.pt"
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

# 클래스 이름
CLASS_NAMES = {
    0: "stamp",
    1: "signature",
    2: "table",
    3: "barcode",
    4: "qrcode",
    5: "hospital_logo"
}

def main():
    print("=" * 60)
    print("학습된 YOLO 모델 테스트")
    print("=" * 60)

    # 모델 로드
    print(f"\n모델 경로: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"오류: 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    print("모델 로드 완료!")

    # 검증 이미지 목록
    val_images = list(VAL_IMAGES_PATH.glob("*.png")) + list(VAL_IMAGES_PATH.glob("*.jpg"))
    print(f"\n검증 이미지 수: {len(val_images)}")

    if len(val_images) == 0:
        print("오류: 검증 이미지가 없습니다.")
        return

    # 랜덤으로 5개 이미지 테스트
    test_images = random.sample(val_images, min(5, len(val_images)))

    print("\n" + "-" * 60)
    print("개별 이미지 테스트 결과")
    print("-" * 60)

    for img_path in test_images:
        print(f"\n이미지: {img_path.name}")
        results = model(str(img_path), verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                print(f"  - {cls_name}: {conf:.3f}")
        else:
            print("  - 검출된 객체 없음")

    # 전체 검증 성능 측정
    print("\n" + "-" * 60)
    print("전체 검증 성능 (validation set)")
    print("-" * 60)

    metrics = model.val(data=str(PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"), verbose=False)

    print(f"\n전체 mAP50: {metrics.box.map50:.4f}")
    print(f"전체 mAP50-95: {metrics.box.map:.4f}")

    print("\n클래스별 성능:")
    for i, ap50 in enumerate(metrics.box.ap50):
        cls_name = CLASS_NAMES.get(i, f"class_{i}")
        print(f"  {cls_name}: mAP50 = {ap50:.4f}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
