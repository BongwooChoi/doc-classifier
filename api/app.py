"""FastAPI 기반 문서 분류 API"""

import sys
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# 문서 클래스 및 키워드 정의
DOCUMENT_CLASSES = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]
ID2LABEL = {i: label for i, label in enumerate(DOCUMENT_CLASSES)}
LABEL2ID = {label: i for i, label in enumerate(DOCUMENT_CLASSES)}

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

# 파일명 매핑 (테스트용)
FILENAME_TO_CLASS = {
    "diagnosis": "진단서",
    "opinion": "소견서",
    "claim": "보험금청구서",
    "insurance": "보험금청구서",
    "admission": "입퇴원확인서",
    "receipt": "의료비영수증",
    "prescription": "처방전"
}

# FastAPI 앱 생성
app = FastAPI(
    title="문서 분류 API",
    description="3단계 파이프라인 기반 의료/보험 문서 분류 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 모델 저장
yolo_detector = None
yolo_classifier = None
layoutlm_model = None
layoutlm_processor = None


class ClassificationResponse(BaseModel):
    """분류 결과 응답 모델"""
    predicted_class: str
    confidence: float
    final_step: int
    processing_time: float
    step1_result: Optional[dict] = None
    step2_result: Optional[dict] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    models_loaded: dict


def get_class_from_filename(filename: str) -> Optional[str]:
    """파일명에서 문서 유형 추출 (테스트용)"""
    for prefix, cls in FILENAME_TO_CLASS.items():
        if filename.lower().startswith(prefix):
            return cls
    return None


def generate_mock_ocr_text(doc_class: str) -> tuple:
    """문서 유형에 맞는 가상 OCR 텍스트 생성"""
    keywords = DOCUMENT_KEYWORDS.get(doc_class, DOCUMENT_KEYWORDS["진단서"])
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


def load_models():
    """모델 로드"""
    global yolo_detector, yolo_classifier, layoutlm_model, layoutlm_processor

    # YOLO 모델 로드
    yolo_model_path = PROJECT_ROOT / "data" / "models" / "yolo_document_layout" / "weights" / "best.pt"
    if yolo_model_path.exists():
        from src.step1_yolo.detector import YOLODetector
        from src.step1_yolo.classifier import YOLOClassifier

        yolo_detector = YOLODetector(
            model_path=str(yolo_model_path),
            confidence_threshold=0.5
        )
        yolo_classifier = YOLOClassifier(
            detector=yolo_detector,
            confidence_threshold=0.7
        )
        print(f"YOLO 모델 로드 완료: {yolo_model_path}")

    # LayoutLM 모델 로드
    layoutlm_model_path = PROJECT_ROOT / "data" / "models" / "layoutlm_classifier" / "best"
    if layoutlm_model_path.exists():
        import torch
        from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

        layoutlm_processor = LayoutLMv3Processor.from_pretrained(
            str(layoutlm_model_path), apply_ocr=False
        )
        layoutlm_model = LayoutLMv3ForSequenceClassification.from_pretrained(
            str(layoutlm_model_path)
        )
        layoutlm_model.eval()
        print(f"LayoutLM 모델 로드 완료: {layoutlm_model_path}")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("모델 로딩 중...")
    load_models()
    print("모델 로딩 완료!")


@app.get("/", response_model=dict)
async def root():
    """API 루트"""
    return {
        "message": "문서 분류 API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "헬스 체크",
            "/classes": "지원 문서 클래스 목록",
            "/classify": "문서 분류 (POST)",
            "/docs": "API 문서 (Swagger UI)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "yolo": yolo_classifier is not None,
            "layoutlm": layoutlm_model is not None
        }
    )


@app.get("/classes", response_model=List[str])
async def get_classes():
    """지원하는 문서 클래스 목록"""
    return DOCUMENT_CLASSES


@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
    use_mock_ocr: bool = Query(
        True,
        description="Mock OCR 사용 여부 (True: 파일명 기반 키워드 생성, False: 실제 OCR)"
    ),
    force_step2: bool = Query(
        False,
        description="Step 2 강제 실행 (Step 1 결과와 무관하게)"
    )
):
    """
    문서 이미지를 업로드하여 분류합니다.

    - **file**: 문서 이미지 파일 (PNG, JPG)
    - **use_mock_ocr**: Mock OCR 사용 여부 (학습 시 Mock OCR 사용)
    - **force_step2**: Step 2 강제 실행 여부
    """
    start_time = time.time()

    # 파일 검증
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="PNG 또는 JPG 파일만 지원합니다.")

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 이미지 로드
        image = Image.open(tmp_path).convert("RGB")
        image_np = np.array(image)

        step1_result = None
        step2_result = None
        final_step = 0
        predicted_class = "unknown"
        confidence = 0.0

        # Step 1: YOLO 분류
        if yolo_classifier is not None:
            step1_result = yolo_classifier.classify(image_np)
            predicted_class = step1_result["predicted_class"]
            confidence = step1_result["confidence"]
            final_step = 1

            # Step 1 결과 정리
            step1_result = {
                "predicted_class": step1_result["predicted_class"],
                "confidence": step1_result["confidence"],
                "requires_step2": step1_result["requires_step2"],
                "layout_features": {
                    k: v for k, v in step1_result["layout_features"].items()
                    if k != "detection_summary"
                }
            }

        # Step 2: LayoutLM 분류 (조건 충족 시)
        needs_step2 = (
            layoutlm_model is not None and
            (force_step2 or (step1_result and step1_result.get("requires_step2", True)))
        )

        if needs_step2:
            import torch

            # OCR 텍스트 준비
            if use_mock_ocr:
                # 파일명에서 문서 유형 추출하여 mock OCR 생성
                doc_class = get_class_from_filename(file.filename)
                if doc_class is None:
                    # 파일명에서 유형을 알 수 없으면 Step 1 결과 사용
                    doc_class = predicted_class if predicted_class != "unknown" else "진단서"
                words, boxes = generate_mock_ocr_text(doc_class)
            else:
                # 실제 OCR 사용 (EasyOCR)
                try:
                    from src.step2_layoutlm.ocr import OCRProcessor
                    ocr = OCRProcessor(engine="easyocr", language="korean", use_gpu=False)
                    ocr_result = ocr.prepare_layoutlm_input(image)
                    words = ocr_result["words"]
                    boxes = ocr_result["boxes"]
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"OCR 처리 오류: {str(e)}")

            # LayoutLM 추론
            encoding = layoutlm_processor(
                image, words, boxes=boxes,
                max_length=512, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = layoutlm_model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = probs.argmax().item()
                pred_label = ID2LABEL[pred_idx]
                pred_confidence = probs[pred_idx].item()

            predicted_class = pred_label
            confidence = pred_confidence
            final_step = 2

            # Step 2 결과 정리
            step2_result = {
                "predicted_class": pred_label,
                "confidence": pred_confidence,
                "all_probabilities": {
                    ID2LABEL[i]: float(p) for i, p in enumerate(probs.tolist())
                },
                "ocr_words_count": len(words),
                "mock_ocr_used": use_mock_ocr
            }

        processing_time = time.time() - start_time

        return ClassificationResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            final_step=final_step,
            processing_time=round(processing_time, 3),
            step1_result=step1_result,
            step2_result=step2_result
        )

    finally:
        # 임시 파일 삭제
        os.unlink(tmp_path)


@app.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    use_mock_ocr: bool = Query(True)
):
    """
    여러 문서 이미지를 일괄 분류합니다.

    - **files**: 문서 이미지 파일들 (PNG, JPG)
    - **use_mock_ocr**: Mock OCR 사용 여부
    """
    results = []
    for file in files:
        try:
            result = await classify_document(file, use_mock_ocr, force_step2=False)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"total": len(files), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
