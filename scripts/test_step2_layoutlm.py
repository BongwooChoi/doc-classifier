#!/usr/bin/env python3
"""Step 2 LayoutLM 분류기 테스트 스크립트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "layoutlm_classifier" / "best"
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

# 문서 클래스
DOCUMENT_CLASSES = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]
ID2LABEL = {i: label for i, label in enumerate(DOCUMENT_CLASSES)}
LABEL2ID = {label: i for i, label in enumerate(DOCUMENT_CLASSES)}

# 문서 유형별 핵심 키워드 (OCR 시뮬레이션용)
DOCUMENT_KEYWORDS = {
    "진단서": ["진단서", "진단명", "상병명", "질병분류", "진단일", "진단코드", "환자성명",
               "상기 환자는", "진단하였음", "의료법", "발급일", "의사", "병원"],
    "소견서": ["소견서", "의료소견", "의학적 소견", "치료소견", "소견", "치료경과",
               "향후 치료계획", "담당의 소견", "상기 환자에 대하여", "병원"],
    "보험금청구서": ["보험금청구서", "청구서", "보험금", "청구금액", "사고내용", "청구인",
                  "계좌번호", "은행", "예금주", "보험계약자", "피보험자", "청구일"],
    "입퇴원확인서": ["입퇴원확인서", "입원확인서", "퇴원확인서", "입원일", "퇴원일",
                  "입원기간", "병실", "진료과", "담당의사", "병원", "확인"],
    "의료비영수증": ["진료비영수증", "영수증", "진료비", "본인부담금", "급여", "비급여",
                  "합계", "수납", "납부금액", "수납일", "병원"],
    "처방전": ["처방전", "처방의약품", "약품명", "투약일수", "용법", "용량", "조제",
              "처방의", "처방일", "의약품", "병원", "약국"]
}

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


def generate_mock_ocr_text(doc_class: str) -> tuple:
    """문서 유형에 맞는 가상 OCR 텍스트 생성"""
    keywords = DOCUMENT_KEYWORDS[doc_class]

    # 해당 문서 키워드를 많이 포함
    selected_keywords = random.sample(keywords, min(len(keywords), random.randint(5, 10)))

    # 공통 텍스트 추가
    common = ["환자명", "홍길동", "성별", "남", "연락처"]
    selected_keywords.extend(random.sample(common, random.randint(1, 3)))

    random.shuffle(selected_keywords)

    # 가상 박스 생성 (LayoutLM 정규화 좌표 0-1000)
    boxes = []
    y_pos = 50
    for text in selected_keywords:
        x_pos = random.randint(50, 300)
        width = min(600, len(text) * 40 + random.randint(10, 50))
        height = 35
        boxes.append([x_pos, y_pos, min(950, x_pos + width), y_pos + height])
        y_pos += random.randint(45, 85)
        if y_pos > 900:
            y_pos = random.randint(50, 200)

    return selected_keywords, boxes


def main():
    print("=" * 70)
    print("Step 2: LayoutLM 문서 분류기 테스트")
    print("=" * 70)

    # 모델 로드
    print(f"\n모델 경로: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"오류: 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return

    print("모델 로드 중...")
    processor = LayoutLMv3Processor.from_pretrained(str(MODEL_PATH), apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(str(MODEL_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"모델 로드 완료! (device: {device})")

    # 검증 이미지 목록
    val_images = list(VAL_IMAGES_PATH.glob("*.png")) + list(VAL_IMAGES_PATH.glob("*.jpg"))
    print(f"검증 이미지 수: {len(val_images)}")

    # 결과 집계
    results_by_type = defaultdict(lambda: {"correct": 0, "incorrect": 0, "predictions": []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    print("\n" + "-" * 70)
    print("분류 테스트 진행 중...")
    print("-" * 70)

    for img_path in sorted(val_images):
        true_label = get_class_from_filename(img_path.stem)
        if true_label == "unknown":
            continue

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # 가상 OCR 텍스트 생성
        words, boxes = generate_mock_ocr_text(true_label)

        # 프로세서로 인코딩
        encoding = processor(
            image, words, boxes=boxes,
            max_length=512, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # 예측
        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            pred_label = ID2LABEL[pred_idx]
            confidence = probs[pred_idx].item()

        # 결과 기록
        is_correct = (pred_label == true_label)
        results_by_type[true_label]["predictions"].append({
            "file": img_path.name,
            "predicted": pred_label,
            "confidence": confidence,
            "correct": is_correct
        })

        if is_correct:
            results_by_type[true_label]["correct"] += 1
        else:
            results_by_type[true_label]["incorrect"] += 1

        confusion_matrix[true_label][pred_label] += 1

    # 오분류만 출력
    print("\n오분류된 샘플:")
    any_error = False
    for doc_type in DOCUMENT_CLASSES:
        for pred in results_by_type[doc_type]["predictions"]:
            if not pred["correct"]:
                any_error = True
                print(f"  {pred['file']}: 정답={doc_type}, 예측={pred['predicted']} ({pred['confidence']:.2f})")

    if not any_error:
        print("  (오분류 없음!)")

    # 문서 유형별 정확도
    print("\n" + "=" * 70)
    print("문서 유형별 정확도")
    print("=" * 70)

    total_correct = 0
    total_count = 0

    for doc_type in DOCUMENT_CLASSES:
        stats = results_by_type[doc_type]
        correct = stats["correct"]
        total = correct + stats["incorrect"]
        total_correct += correct
        total_count += total
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"  {doc_type}: {correct}/{total} ({accuracy:.1f}%)")

    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    print(f"\n전체 정확도: {total_correct}/{total_count} ({overall_accuracy:.1f}%)")

    # Step 1과 비교
    print("\n" + "=" * 70)
    print("Step 1 vs Step 2 비교")
    print("=" * 70)
    print("  Step 1 (YOLO): 레이아웃 기반 - 진단서/소견서 구분 불가")
    print("  Step 2 (LayoutLM): 텍스트 기반 - 진단서/소견서 구분 가능")

    # 진단서/소견서 구분 확인
    diagnosis_correct = results_by_type["진단서"]["correct"]
    diagnosis_total = diagnosis_correct + results_by_type["진단서"]["incorrect"]
    opinion_correct = results_by_type["소견서"]["correct"]
    opinion_total = opinion_correct + results_by_type["소견서"]["incorrect"]

    print(f"\n  진단서 정확도: {diagnosis_correct}/{diagnosis_total}")
    print(f"  소견서 정확도: {opinion_correct}/{opinion_total}")

    if diagnosis_correct == diagnosis_total and opinion_correct == opinion_total:
        print("\n  => Step 2에서 진단서/소견서 완벽 구분 성공!")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
