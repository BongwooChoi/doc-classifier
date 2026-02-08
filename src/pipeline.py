"""3단계 파이프라인 기반 문서 분류 시스템"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import yaml

from .preprocessor import Deskewer, DocumentDetector
from .step1_yolo import YOLODetector, YOLOClassifier
from .step2_layoutlm import OCRProcessor, LayoutLMClassifier
from .step3_vlm import VLMHandler
from .utils.logger import get_logger, LoggerMixin
from .utils.image_utils import load_image


@dataclass
class ClassificationResult:
    """문서 분류 결과"""
    predicted_class: str
    confidence: float
    final_step: int  # 최종 분류가 결정된 단계 (1, 2, 또는 3)

    step1_result: Optional[Dict] = None
    step2_result: Optional[Dict] = None
    step3_result: Optional[Dict] = None

    preprocessing_info: Dict = field(default_factory=dict)
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "final_step": self.final_step,
            "step1_result": self.step1_result,
            "step2_result": self.step2_result,
            "step3_result": self.step3_result,
            "preprocessing_info": self.preprocessing_info,
            "processing_time": self.processing_time
        }


class DocumentClassificationPipeline(LoggerMixin):
    """3단계 문서 분류 파이프라인

    Step 1: YOLO 기반 레이아웃 검출 및 1차 분류
    Step 2: LayoutLM 기반 정밀 분류
    Step 3: VLM 기반 예외 처리 (신뢰도 미달 시)
    """

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
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        enable_preprocessing: bool = True,
        enable_step1: bool = True,
        enable_step2: bool = True,
        enable_step3: bool = True
    ):
        """
        Args:
            config_path: 설정 파일 경로
            config: 설정 딕셔너리 (config_path보다 우선)
            enable_preprocessing: 전처리 활성화 여부
            enable_step1: Step 1 (YOLO) 활성화 여부
            enable_step2: Step 2 (LayoutLM) 활성화 여부
            enable_step3: Step 3 (VLM) 활성화 여부
        """
        self.config = config or self._load_config(config_path)

        self.enable_preprocessing = enable_preprocessing
        self.enable_step1 = enable_step1
        self.enable_step2 = enable_step2
        self.enable_step3 = enable_step3

        self.deskewer = None
        self.document_detector = None
        self.yolo_detector = None
        self.yolo_classifier = None
        self.ocr_processor = None
        self.layoutlm_classifier = None
        self.vlm_handler = None

        self._init_components()

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """설정 파일 로드"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        self.logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return {}

    def _init_components(self) -> None:
        """컴포넌트 초기화"""
        preproc_config = self.config.get("preprocessing", {})

        if self.enable_preprocessing:
            deskew_config = preproc_config.get("deskew", {})
            if deskew_config.get("enabled", True):
                self.deskewer = Deskewer(
                    max_angle=deskew_config.get("max_angle", 15)
                )

            doc_config = preproc_config.get("document_detection", {})
            if doc_config.get("enabled", True):
                self.document_detector = DocumentDetector(
                    min_area_ratio=doc_config.get("min_area_ratio", 0.3)
                )

        if self.enable_step1:
            yolo_config = self.config.get("yolo", {})
            self.yolo_detector = YOLODetector(
                model_path=yolo_config.get("model_path"),
                confidence_threshold=yolo_config.get("confidence_threshold", 0.5),
                iou_threshold=yolo_config.get("iou_threshold", 0.45)
            )
            self.yolo_classifier = YOLOClassifier(
                detector=self.yolo_detector,
                confidence_threshold=yolo_config.get("confidence_threshold", 0.5)
            )

        if self.enable_step2:
            ocr_config = self.config.get("ocr", {})
            self.ocr_processor = OCRProcessor(
                engine=ocr_config.get("engine", "paddleocr"),
                language=ocr_config.get("language", "korean"),
                use_gpu=ocr_config.get("use_gpu", False)
            )

            layoutlm_config = self.config.get("layoutlm", {})
            self.layoutlm_classifier = LayoutLMClassifier(
                model_path=layoutlm_config.get("model_path"),
                model_name=layoutlm_config.get("model_name", "microsoft/layoutlmv3-base"),
                ocr_processor=self.ocr_processor,
                max_seq_length=layoutlm_config.get("max_seq_length", 512),
                confidence_threshold=layoutlm_config.get("confidence_threshold", 0.7)
            )

        if self.enable_step3:
            vlm_config = self.config.get("vlm", {})
            provider = vlm_config.get("provider", "openai")
            provider_config = vlm_config.get(provider, {})

            self.vlm_handler = VLMHandler(
                provider=provider,
                model=provider_config.get("model"),
                max_tokens=provider_config.get("max_tokens", 500)
            )

        self.logger.info("파이프라인 컴포넌트 초기화 완료")

    def preprocess(self, image: np.ndarray) -> tuple:
        """이미지 전처리

        Args:
            image: 입력 이미지

        Returns:
            (전처리된 이미지, 전처리 정보) 튜플
        """
        info = {}
        processed = image.copy()

        if self.deskewer:
            processed, angle = self.deskewer(processed)
            info["deskew_angle"] = angle

        if self.document_detector:
            processed, contour = self.document_detector(processed)
            info["document_detected"] = contour is not None
            if contour is not None:
                info["document_bbox"] = self.document_detector.get_bounding_box(contour)

        return processed, info

    def classify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        skip_preprocessing: bool = False,
        force_all_steps: bool = False
    ) -> ClassificationResult:
        """문서 분류 실행

        Args:
            image: 입력 이미지
            skip_preprocessing: 전처리 건너뛰기
            force_all_steps: 모든 단계 강제 실행

        Returns:
            분류 결과
        """
        import time
        start_time = time.time()

        if isinstance(image, (str, Path)):
            image = load_image(image, mode="cv2")
        elif isinstance(image, Image.Image):
            image = np.array(image)

        preprocessing_info = {}
        if self.enable_preprocessing and not skip_preprocessing:
            image, preprocessing_info = self.preprocess(image)

        step1_result = None
        step2_result = None
        step3_result = None
        final_step = 0
        predicted_class = "unknown"
        confidence = 0.0

        if self.enable_step1:
            self.logger.info("Step 1: YOLO 분류 실행")
            step1_result = self.yolo_classifier(image)
            predicted_class = step1_result["predicted_class"]
            confidence = step1_result["confidence"]
            final_step = 1

            if not step1_result["requires_step2"] and not force_all_steps:
                self.logger.info(f"Step 1에서 분류 완료: {predicted_class}")
                return ClassificationResult(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    final_step=1,
                    step1_result=step1_result,
                    preprocessing_info=preprocessing_info,
                    processing_time=time.time() - start_time
                )

        if self.enable_step2:
            self.logger.info("Step 2: LayoutLM 분류 실행")
            step2_result = self.layoutlm_classifier(
                image,
                step1_result=step1_result
            )
            predicted_class = step2_result["predicted_class"]
            confidence = step2_result["confidence"]
            final_step = 2

            if not step2_result["requires_vlm"] and not force_all_steps:
                self.logger.info(f"Step 2에서 분류 완료: {predicted_class}")
                return ClassificationResult(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    final_step=2,
                    step1_result=step1_result,
                    step2_result=step2_result,
                    preprocessing_info=preprocessing_info,
                    processing_time=time.time() - start_time
                )

        if self.enable_step3:
            self.logger.info("Step 3: VLM 분류 실행")
            step3_result = self.vlm_handler(
                image,
                step1_result=step1_result,
                step2_result=step2_result
            )
            predicted_class = step3_result["predicted_class"]
            confidence = step3_result.get("confidence", 0.9)
            final_step = 3

            self.logger.info(f"Step 3에서 분류 완료: {predicted_class}")

        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            final_step=final_step,
            step1_result=step1_result,
            step2_result=step2_result,
            step3_result=step3_result,
            preprocessing_info=preprocessing_info,
            processing_time=time.time() - start_time
        )

    def batch_classify(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        show_progress: bool = True
    ) -> List[ClassificationResult]:
        """여러 이미지 일괄 분류

        Args:
            images: 이미지 리스트
            show_progress: 진행 상황 표시 여부

        Returns:
            분류 결과 리스트
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                images = tqdm(images, desc="문서 분류 중")
            except ImportError:
                pass

        for image in images:
            result = self.classify(image)
            results.append(result)

        return results

    def get_statistics(
        self,
        results: List[ClassificationResult]
    ) -> Dict:
        """분류 결과 통계

        Args:
            results: 분류 결과 리스트

        Returns:
            통계 딕셔너리
        """
        total = len(results)

        class_counts = {cls: 0 for cls in self.DOCUMENT_CLASSES}
        step_counts = {1: 0, 2: 0, 3: 0}
        total_time = 0.0
        confidences = []

        for result in results:
            if result.predicted_class in class_counts:
                class_counts[result.predicted_class] += 1
            step_counts[result.final_step] += 1
            if result.processing_time:
                total_time += result.processing_time
            confidences.append(result.confidence)

        return {
            "total_documents": total,
            "class_distribution": class_counts,
            "step_distribution": {
                f"step{k}": v for k, v in step_counts.items()
            },
            "vlm_usage_rate": step_counts[3] / total if total > 0 else 0,
            "average_processing_time": total_time / total if total > 0 else 0,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0
        }

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> ClassificationResult:
        """classify 메서드 호출"""
        return self.classify(image)
