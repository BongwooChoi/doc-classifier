"""OCR 처리 모듈"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..utils.logger import LoggerMixin


class OCRProcessor(LoggerMixin):
    """OCR 텍스트 및 바운딩 박스 추출"""

    def __init__(
        self,
        engine: str = "paddleocr",
        language: str = "korean",
        use_gpu: bool = False
    ):
        """
        Args:
            engine: OCR 엔진 ("paddleocr" 또는 "easyocr")
            language: 인식 언어
            use_gpu: GPU 사용 여부
        """
        self.engine = engine
        self.language = language
        self.use_gpu = use_gpu
        self.ocr = None

    def _init_paddleocr(self) -> None:
        """PaddleOCR 초기화"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr 패키지를 설치해주세요: pip install paddleocr paddlepaddle")

        lang_map = {
            "korean": "korean",
            "ko": "korean",
            "english": "en",
            "en": "en",
            "chinese": "ch",
            "ch": "ch"
        }

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang_map.get(self.language, "korean"),
            use_gpu=self.use_gpu,
            show_log=False
        )
        self.logger.info("PaddleOCR 초기화 완료")

    def _init_easyocr(self) -> None:
        """EasyOCR 초기화"""
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr 패키지를 설치해주세요: pip install easyocr")

        lang_map = {
            "korean": ["ko", "en"],
            "ko": ["ko", "en"],
            "english": ["en"],
            "en": ["en"],
            "chinese": ["ch_sim", "en"],
            "ch": ["ch_sim", "en"]
        }

        self.ocr = easyocr.Reader(
            lang_map.get(self.language, ["ko", "en"]),
            gpu=self.use_gpu
        )
        self.logger.info("EasyOCR 초기화 완료")

    def load_engine(self) -> None:
        """OCR 엔진 로드"""
        if self.engine == "paddleocr":
            self._init_paddleocr()
        elif self.engine == "easyocr":
            self._init_easyocr()
        else:
            raise ValueError(f"지원하지 않는 OCR 엔진입니다: {self.engine}")

    def extract(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict:
        """이미지에서 텍스트 및 위치 정보 추출

        Args:
            image: 입력 이미지

        Returns:
            추출 결과:
            {
                "texts": List[str],
                "boxes": List[[x1, y1, x2, y2]],
                "confidences": List[float],
                "words": List[Dict],  # 개별 단어 정보
                "full_text": str
            }
        """
        if self.ocr is None:
            self.load_engine()

        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_array = np.array(Image.open(image_path).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"))
        else:
            image_array = image

        if self.engine == "paddleocr":
            return self._extract_paddleocr(image_array)
        else:
            return self._extract_easyocr(image_array)

    def _extract_paddleocr(self, image: np.ndarray) -> Dict:
        """PaddleOCR로 텍스트 추출"""
        result = self.ocr.ocr(image, cls=True)

        texts = []
        boxes = []
        confidences = []
        words = []

        if result and result[0]:
            for line in result[0]:
                box_points = line[0]
                text, conf = line[1]

                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                texts.append(text)
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

                words.append({
                    "text": text,
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "polygon": box_points
                })

        full_text = " ".join(texts)

        self.logger.debug(f"OCR 완료: {len(texts)}개 텍스트 영역")

        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences,
            "words": words,
            "full_text": full_text
        }

    def _extract_easyocr(self, image: np.ndarray) -> Dict:
        """EasyOCR로 텍스트 추출"""
        result = self.ocr.readtext(image)

        texts = []
        boxes = []
        confidences = []
        words = []

        for detection in result:
            box_points, text, conf = detection

            x_coords = [p[0] for p in box_points]
            y_coords = [p[1] for p in box_points]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            texts.append(text)
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

            words.append({
                "text": text,
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "polygon": box_points
            })

        full_text = " ".join(texts)

        self.logger.debug(f"OCR 완료: {len(texts)}개 텍스트 영역")

        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences,
            "words": words,
            "full_text": full_text
        }

    def normalize_boxes(
        self,
        boxes: List[List[float]],
        image_size: Tuple[int, int],
        target_size: int = 1000
    ) -> List[List[int]]:
        """바운딩 박스를 정규화된 좌표로 변환 (LayoutLM용)

        Args:
            boxes: 원본 바운딩 박스 리스트
            image_size: 원본 이미지 크기 (width, height)
            target_size: 정규화 목표 크기 (기본 1000)

        Returns:
            정규화된 바운딩 박스 리스트
        """
        width, height = image_size
        normalized_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            normalized_boxes.append([
                int(x1 / width * target_size),
                int(y1 / height * target_size),
                int(x2 / width * target_size),
                int(y2 / height * target_size)
            ])

        return normalized_boxes

    def prepare_layoutlm_input(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict:
        """LayoutLM 입력 형식으로 데이터 준비

        Args:
            image: 입력 이미지

        Returns:
            LayoutLM 입력 형식 딕셔너리
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        ocr_result = self.extract(image)

        width, height = pil_image.size
        normalized_boxes = self.normalize_boxes(
            ocr_result["boxes"],
            (width, height)
        )

        return {
            "image": pil_image,
            "words": ocr_result["texts"],
            "boxes": normalized_boxes,
            "original_boxes": ocr_result["boxes"],
            "image_size": (width, height)
        }

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict:
        """extract 메서드 호출"""
        return self.extract(image)
