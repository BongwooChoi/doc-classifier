"""LayoutLM 기반 정밀 문서 분류 모듈"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from .ocr import OCRProcessor
from ..utils.logger import LoggerMixin


class LayoutLMClassifier(LoggerMixin):
    """LayoutLMv3 기반 문서 정밀 분류기"""

    DOCUMENT_CLASSES = [
        "진단서",
        "소견서",
        "보험금청구서",
        "입퇴원확인서",
        "의료비영수증",
        "처방전"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "microsoft/layoutlmv3-base",
        ocr_processor: Optional[OCRProcessor] = None,
        max_seq_length: int = 512,
        confidence_threshold: float = 0.7,
        device: str = "auto"
    ):
        """
        Args:
            model_path: 학습된 모델 경로 (None이면 사전학습 모델 사용)
            model_name: 사전학습 모델 이름
            ocr_processor: OCR 처리기 인스턴스
            max_seq_length: 최대 시퀀스 길이
            confidence_threshold: VLM으로 전달할 신뢰도 임계값
            device: 실행 장치
        """
        self.model_path = model_path
        self.model_name = model_name
        self.ocr_processor = ocr_processor or OCRProcessor()
        self.max_seq_length = max_seq_length
        self.confidence_threshold = confidence_threshold

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None
        self.id2label = {i: label for i, label in enumerate(self.DOCUMENT_CLASSES)}
        self.label2id = {label: i for i, label in enumerate(self.DOCUMENT_CLASSES)}

    def load_model(self) -> None:
        """LayoutLM 모델 및 프로세서 로드"""
        try:
            from transformers import (
                LayoutLMv3ForSequenceClassification,
                LayoutLMv3Processor
            )
        except ImportError:
            raise ImportError(
                "transformers 패키지를 설치해주세요: pip install transformers"
            )

        if self.model_path and Path(self.model_path).exists():
            self.logger.info(f"학습된 모델 로드: {self.model_path}")
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_path, apply_ocr=False
            )
        else:
            self.logger.info(f"사전학습 모델 로드: {self.model_name}")
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.DOCUMENT_CLASSES),
                id2label=self.id2label,
                label2id=self.label2id
            )
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name, apply_ocr=False
            )

        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"LayoutLM 모델 로드 완료 (device: {self.device})")

    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        ocr_result: Optional[Dict] = None
    ) -> Dict:
        """입력 데이터 전처리

        Args:
            image: 입력 이미지
            ocr_result: 미리 계산된 OCR 결과 (선택)

        Returns:
            모델 입력 텐서 딕셔너리
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.convert("RGB")

        if ocr_result is None:
            ocr_data = self.ocr_processor.prepare_layoutlm_input(pil_image)
        else:
            width, height = pil_image.size
            normalized_boxes = self.ocr_processor.normalize_boxes(
                ocr_result["boxes"],
                (width, height)
            )
            ocr_data = {
                "image": pil_image,
                "words": ocr_result["texts"],
                "boxes": normalized_boxes
            }

        words = ocr_data["words"]
        boxes = ocr_data["boxes"]

        if not words:
            words = ["[EMPTY]"]
            boxes = [[0, 0, 0, 0]]

        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def classify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        ocr_result: Optional[Dict] = None,
        step1_result: Optional[Dict] = None
    ) -> Dict:
        """문서 정밀 분류

        Args:
            image: 입력 이미지
            ocr_result: 미리 계산된 OCR 결과 (선택)
            step1_result: Step 1 분류 결과 (선택)

        Returns:
            분류 결과 딕셔너리:
            {
                "predicted_class": str,
                "confidence": float,
                "all_probabilities": Dict[str, float],
                "requires_vlm": bool,
                "step1_class": Optional[str],
                "step1_confidence": Optional[float]
            }
        """
        if self.model is None:
            self.load_model()

        inputs = self.preprocess(image, ocr_result)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        probs_np = probabilities.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        predicted_class = self.id2label[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        all_probabilities = {
            self.id2label[i]: float(prob)
            for i, prob in enumerate(probs_np)
        }

        requires_vlm = confidence < self.confidence_threshold

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "requires_vlm": requires_vlm
        }

        if step1_result:
            result["step1_class"] = step1_result.get("predicted_class")
            result["step1_confidence"] = step1_result.get("confidence")

            if result["step1_class"] == predicted_class:
                result["confidence"] = min(1.0, confidence * 1.1)

        self.logger.info(
            f"정밀 분류 결과: {predicted_class} (신뢰도: {confidence:.2f})"
        )

        if requires_vlm:
            self.logger.info("VLM 처리 필요 (신뢰도 미달)")

        return result

    def get_top_predictions(
        self,
        classification_result: Dict,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """상위 K개 예측 결과 반환

        Args:
            classification_result: classify() 메서드의 반환값
            top_k: 반환할 상위 결과 수

        Returns:
            [(클래스명, 확률), ...] 리스트
        """
        all_probs = classification_result.get("all_probabilities", {})
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_k]

    def batch_classify(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        batch_size: int = 8
    ) -> List[Dict]:
        """여러 이미지 일괄 분류

        Args:
            images: 이미지 리스트
            batch_size: 배치 크기

        Returns:
            분류 결과 리스트
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            self.logger.debug(
                f"배치 {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size} 처리 중..."
            )

            for image in batch:
                result = self.classify(image)
                results.append(result)

        return results

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        ocr_result: Optional[Dict] = None
    ) -> Dict:
        """classify 메서드 호출"""
        return self.classify(image, ocr_result)
