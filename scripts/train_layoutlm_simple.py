#!/usr/bin/env python3
"""LayoutLM 간단 학습 스크립트 - 독립 실행 가능"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer
)

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "images"
OUTPUT_PATH = PROJECT_ROOT / "data" / "models" / "layoutlm_classifier"

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


def prepare_dataset(split: str = "train") -> list:
    """데이터셋 준비"""
    images_dir = IMAGES_PATH / split
    data = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        doc_class = get_class_from_filename(img_path.stem)
        if doc_class == "unknown":
            continue

        label_id = LABEL2ID[doc_class]

        # 가상 OCR 텍스트 생성
        words, boxes = generate_mock_ocr_text(doc_class)

        data.append({
            "image_path": str(img_path),
            "words": words,
            "boxes": boxes,
            "label": label_id
        })

    return data


class LayoutLMDataset(Dataset):
    """LayoutLM 학습용 데이터셋"""

    def __init__(self, data: list, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 이미지 로드
        image = Image.open(item["image_path"]).convert("RGB")

        # 프로세서로 인코딩
        encoding = self.processor(
            image,
            item["words"],
            boxes=item["boxes"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 배치 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(item["label"])

        return encoding


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    print("=" * 60)
    print("LayoutLM 문서 분류기 학습")
    print("=" * 60)

    # 데이터 준비
    print("\n데이터 준비 중...")
    train_data = prepare_dataset("train")
    val_data = prepare_dataset("val")

    print(f"학습 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(val_data)}개")

    # 클래스별 분포 확인
    train_dist = {}
    for item in train_data:
        label = ID2LABEL[item["label"]]
        train_dist[label] = train_dist.get(label, 0) + 1

    print("\n학습 데이터 분포:")
    for cls, cnt in sorted(train_dist.items()):
        print(f"  {cls}: {cnt}개")

    # 프로세서 및 모델 로드
    print("\n모델 로드 중...")
    model_name = "microsoft/layoutlmv3-base"

    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(DOCUMENT_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 데이터셋 생성
    train_dataset = LayoutLMDataset(train_data, processor)
    val_dataset = LayoutLMDataset(val_data, processor)

    # 출력 디렉토리 생성
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    best_model_path = OUTPUT_PATH / "best"

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=30,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_PATH / "logs"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0
    )

    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 학습
    print("\n학습 시작...")
    print("-" * 60)

    trainer.train()

    # 모델 저장
    print("\n모델 저장 중...")
    trainer.save_model(str(best_model_path))
    processor.save_pretrained(str(best_model_path))

    # 최종 평가
    print("\n최종 평가...")
    eval_results = trainer.evaluate()
    print(f"검증 정확도: {eval_results['eval_accuracy']:.4f}")

    # 클래스별 예측 테스트
    print("\n" + "-" * 60)
    print("샘플 예측 테스트")
    print("-" * 60)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for doc_class in DOCUMENT_CLASSES[:3]:
        sample_data = [d for d in val_data if ID2LABEL[d["label"]] == doc_class]
        if sample_data:
            item = sample_data[0]
            image = Image.open(item["image_path"]).convert("RGB")

            encoding = processor(
                image, item["words"], boxes=item["boxes"],
                max_length=512, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = probs.argmax().item()
                pred_label = ID2LABEL[pred_idx]
                conf = probs[pred_idx].item()

            status = "✓" if pred_label == doc_class else "✗"
            print(f"  {status} 정답: {doc_class}, 예측: {pred_label} ({conf:.2f})")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델 저장 위치: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
