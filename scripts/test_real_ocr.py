#!/usr/bin/env python3
"""실제 OCR(EasyOCR) vs Mock OCR 비교 테스트 스크립트

합성 이미지(validation set)에 대해 실제 EasyOCR을 사용하여 텍스트를 추출하고,
Mock OCR과 비교하여 정확도를 측정합니다.

주의: 모델은 Mock OCR(문서 유형별 키워드)로 학습되었으므로,
실제 OCR 결과는 정확도가 낮을 수 있습니다. 이는 합성 데이터의 한계이며
파이프라인 결함이 아닙니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

from src.step2_layoutlm.ocr import OCRProcessor

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "layoutlm_classifier" / "best"
VAL_IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"

# 문서 클래스
DOCUMENT_CLASSES = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]
ID2LABEL = {i: label for i, label in enumerate(DOCUMENT_CLASSES)}
LABEL2ID = {label: i for i, label in enumerate(DOCUMENT_CLASSES)}

# 문서 유형별 핵심 키워드 (Mock OCR용)
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
    selected_keywords = random.sample(keywords, min(len(keywords), random.randint(5, 10)))
    common = ["환자명", "홍길동", "성별", "남", "연락처"]
    selected_keywords.extend(random.sample(common, random.randint(1, 3)))
    random.shuffle(selected_keywords)

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


def classify_image(model, processor, image, words, boxes, device):
    """이미지 분류 실행"""
    encoding = processor(
        image, words, boxes=boxes,
        max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        pred_label = ID2LABEL[pred_idx]
        confidence = probs[pred_idx].item()

    return pred_label, confidence, probs


def main():
    print("=" * 70)
    print("실제 OCR(EasyOCR) vs Mock OCR 비교 테스트")
    print("=" * 70)

    # 모델 로드
    print(f"\n모델 경로: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"오류: 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return

    print("LayoutLM 모델 로드 중...")
    processor = LayoutLMv3Processor.from_pretrained(str(MODEL_PATH), apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(str(MODEL_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"모델 로드 완료! (device: {device})")

    # EasyOCR 초기화
    print("\nEasyOCR 초기화 중 (첫 실행 시 모델 다운로드 ~100MB)...")
    ocr = OCRProcessor(engine="easyocr", language="korean", use_gpu=False)
    ocr.load_engine()
    print("EasyOCR 초기화 완료!")

    # 검증 이미지 목록
    val_images = sorted(
        list(VAL_IMAGES_PATH.glob("*.png")) + list(VAL_IMAGES_PATH.glob("*.jpg"))
    )
    print(f"\n검증 이미지 수: {len(val_images)}")

    # 결과 집계
    mock_results = defaultdict(lambda: {"correct": 0, "total": 0})
    real_results = defaultdict(lambda: {"correct": 0, "total": 0})
    ocr_samples = defaultdict(list)  # 디버깅용 OCR 텍스트 샘플

    print("\n" + "-" * 70)
    print("테스트 진행 중...")
    print("-" * 70)

    start_time = time.time()
    skipped = 0

    for i, img_path in enumerate(val_images):
        true_label = get_class_from_filename(img_path.stem)
        if true_label == "unknown":
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")

        # --- Mock OCR 분류 ---
        mock_words, mock_boxes = generate_mock_ocr_text(true_label)
        mock_pred, mock_conf, _ = classify_image(
            model, processor, image, mock_words, mock_boxes, device
        )
        mock_results[true_label]["total"] += 1
        if mock_pred == true_label:
            mock_results[true_label]["correct"] += 1

        # --- 실제 OCR 분류 ---
        try:
            ocr_result = ocr.prepare_layoutlm_input(image)
            real_words = ocr_result["words"]
            real_boxes = ocr_result["boxes"]
        except Exception as e:
            print(f"  [OCR 오류] {img_path.name}: {e}")
            real_results[true_label]["total"] += 1
            continue

        # OCR 결과가 비어있으면 빈 단어로 처리
        if not real_words:
            real_words = [""]
            real_boxes = [[0, 0, 0, 0]]

        # OCR 텍스트 샘플 저장 (유형별 최대 2개)
        if len(ocr_samples[true_label]) < 2:
            ocr_samples[true_label].append({
                "file": img_path.name,
                "words": real_words[:15],  # 처음 15개만
                "word_count": len(real_words)
            })

        real_pred, real_conf, _ = classify_image(
            model, processor, image, real_words, real_boxes, device
        )
        real_results[true_label]["total"] += 1
        if real_pred == true_label:
            real_results[true_label]["correct"] += 1

        # 진행 상황 표시
        progress = i + 1
        status = "O" if real_pred == true_label else "X"
        print(f"  [{progress}/{len(val_images)}] {img_path.name}: "
              f"정답={true_label}, Mock={mock_pred}({mock_conf:.2f}), "
              f"Real={real_pred}({real_conf:.2f}) [{status}] "
              f"(OCR words: {len(real_words)})")

    elapsed = time.time() - start_time

    # === 결과 요약 ===
    print("\n" + "=" * 70)
    print("OCR 텍스트 샘플 (디버깅용)")
    print("=" * 70)
    for doc_type in DOCUMENT_CLASSES:
        samples = ocr_samples.get(doc_type, [])
        if samples:
            print(f"\n  [{doc_type}]")
            for s in samples:
                words_str = ", ".join(s["words"][:10])
                print(f"    {s['file']} ({s['word_count']}개 단어)")
                print(f"    OCR: {words_str}")

    print("\n" + "=" * 70)
    print("문서 유형별 정확도 비교")
    print("=" * 70)
    print(f"  {'문서 유형':<12} {'Mock OCR':<18} {'Real OCR (EasyOCR)':<18}")
    print(f"  {'-'*12} {'-'*18} {'-'*18}")

    mock_total_correct = 0
    mock_total_count = 0
    real_total_correct = 0
    real_total_count = 0

    for doc_type in DOCUMENT_CLASSES:
        mc = mock_results[doc_type]["correct"]
        mt = mock_results[doc_type]["total"]
        rc = real_results[doc_type]["correct"]
        rt = real_results[doc_type]["total"]

        mock_total_correct += mc
        mock_total_count += mt
        real_total_correct += rc
        real_total_count += rt

        mock_acc = f"{mc}/{mt} ({mc/mt*100:.0f}%)" if mt > 0 else "N/A"
        real_acc = f"{rc}/{rt} ({rc/rt*100:.0f}%)" if rt > 0 else "N/A"
        print(f"  {doc_type:<12} {mock_acc:<18} {real_acc:<18}")

    print(f"  {'-'*12} {'-'*18} {'-'*18}")
    mock_overall = f"{mock_total_correct}/{mock_total_count} ({mock_total_correct/mock_total_count*100:.1f}%)" if mock_total_count > 0 else "N/A"
    real_overall = f"{real_total_correct}/{real_total_count} ({real_total_correct/real_total_count*100:.1f}%)" if real_total_count > 0 else "N/A"
    print(f"  {'전체':<12} {mock_overall:<18} {real_overall:<18}")

    print(f"\n  처리 시간: {elapsed:.1f}초 ({elapsed/len(val_images):.2f}초/이미지)")
    if skipped > 0:
        print(f"  건너뛴 이미지: {skipped}개 (파일명에서 유형 미확인)")

    print("\n" + "=" * 70)
    print("참고사항")
    print("=" * 70)
    print("  - 모델은 Mock OCR(파일명 기반 키워드)로 학습되었습니다.")
    print("  - 실제 OCR 결과와 학습 데이터 간 도메인 차이가 있을 수 있습니다.")
    print("  - 합성 이미지의 텍스트 품질에 따라 OCR 정확도가 달라집니다.")
    print("  - 실제 의료 문서에서는 더 나은 OCR 결과를 기대할 수 있습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
