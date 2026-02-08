"""YOLO 기반 1차 문서 분류 모듈"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image

from .detector import YOLODetector
from ..utils.logger import LoggerMixin


class YOLOClassifier(LoggerMixin):
    """YOLO 검출 결과 기반 1차 문서 분류기"""

    DOCUMENT_CLASSES = [
        "진단서",
        "소견서",
        "보험금청구서",
        "입퇴원확인서",
        "의료비영수증",
        "처방전"
    ]

    CLASSIFICATION_RULES = {
        "진단서": {
            "required": ["stamp", "hospital_logo"],
            "optional": ["signature"],
            "forbidden": ["table", "barcode", "qrcode"],
            "table_ratio_range": (0.0, 0.1),
            "weight": 1.0
        },
        "소견서": {
            "required": ["stamp", "hospital_logo"],
            "optional": ["signature"],
            "forbidden": ["table", "barcode", "qrcode"],
            "table_ratio_range": (0.0, 0.1),
            "weight": 0.95
        },
        "보험금청구서": {
            "required": ["table", "barcode"],
            "optional": ["stamp", "signature", "hospital_logo"],
            "forbidden": ["qrcode"],
            "table_ratio_range": (0.2, 0.6),
            "weight": 1.0
        },
        "입퇴원확인서": {
            "required": ["stamp", "hospital_logo", "table"],
            "optional": [],
            "forbidden": ["barcode", "qrcode"],
            "table_ratio_range": (0.1, 0.5),
            "weight": 1.0
        },
        "의료비영수증": {
            "required": ["table", "barcode"],
            "optional": ["stamp", "hospital_logo"],
            "forbidden": ["qrcode"],
            "table_ratio_range": (0.3, 0.8),
            "weight": 1.0
        },
        "처방전": {
            "required": ["table", "qrcode"],
            "optional": ["stamp", "hospital_logo", "signature"],
            "forbidden": ["barcode"],
            "table_ratio_range": (0.2, 0.5),
            "weight": 1.0
        }
    }

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            detector: YOLO 검출기 인스턴스 (None이면 새로 생성)
            model_path: YOLO 모델 경로
            confidence_threshold: 분류 신뢰도 임계값
        """
        self.detector = detector or YOLODetector(model_path=model_path)
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        detection_result: Optional[Dict] = None
    ) -> Dict:
        """문서 1차 분류

        Args:
            image: 입력 이미지
            detection_result: 미리 계산된 검출 결과 (선택)

        Returns:
            분류 결과 딕셔너리:
            {
                "predicted_class": str,
                "confidence": float,
                "all_scores": Dict[str, float],
                "layout_features": Dict,
                "detection_result": Dict,
                "requires_step2": bool
            }
        """
        if detection_result is None:
            detection_result = self.detector.detect(image)

        features = self.detector.get_layout_features(detection_result)

        scores = self._calculate_scores(features)

        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            predicted_class = sorted_scores[0][0]
            confidence = sorted_scores[0][1]
        else:
            predicted_class = "unknown"
            confidence = 0.0

        requires_step2 = confidence < self.confidence_threshold

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_scores": scores,
            "layout_features": features,
            "detection_result": detection_result,
            "requires_step2": requires_step2
        }

        self.logger.info(
            f"1차 분류 결과: {predicted_class} (신뢰도: {confidence:.2f})"
        )

        return result

    def _calculate_scores(self, features: Dict) -> Dict[str, float]:
        """레이아웃 특징 기반 분류 점수 계산

        Args:
            features: 레이아웃 특징 딕셔너리

        Returns:
            각 문서 유형별 점수 딕셔너리
        """
        scores = {}

        for doc_class, rules in self.CLASSIFICATION_RULES.items():
            score = 0.0
            max_score = 0.0
            penalty = 0.0

            # 필수 요소 체크
            required_count = 0
            for required in rules["required"]:
                max_score += 1.0
                if features.get(f"has_{required}", False):
                    score += 1.0
                    required_count += 1

            # 필수 요소가 하나도 없으면 점수 크게 감소
            if len(rules["required"]) > 0 and required_count == 0:
                penalty += 0.5

            # 선택 요소 체크
            for optional in rules["optional"]:
                max_score += 0.3
                if features.get(f"has_{optional}", False):
                    score += 0.3

            # 금지 요소 체크 (있으면 감점)
            for forbidden in rules.get("forbidden", []):
                if features.get(f"has_{forbidden}", False):
                    penalty += 0.3

            # 테이블 비율 체크
            table_ratio = features.get("table_area_ratio", 0.0)
            min_ratio, max_ratio = rules["table_ratio_range"]
            max_score += 0.4

            if min_ratio <= table_ratio <= max_ratio:
                score += 0.4
            elif table_ratio < min_ratio:
                score += 0.2 * (1 - (min_ratio - table_ratio) / max(min_ratio, 0.01))
            else:
                score += 0.2 * max(0, 1 - (table_ratio - max_ratio) / max(max_ratio, 0.01))

            if max_score > 0:
                normalized_score = (score / max_score) * rules["weight"] - penalty
                scores[doc_class] = max(0.0, min(1.0, normalized_score))

        return scores

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
            [(클래스명, 신뢰도), ...] 리스트
        """
        all_scores = classification_result.get("all_scores", {})
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def batch_classify(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]]
    ) -> List[Dict]:
        """여러 이미지 일괄 분류

        Args:
            images: 이미지 리스트

        Returns:
            분류 결과 리스트
        """
        results = []
        for i, image in enumerate(images):
            self.logger.debug(f"이미지 {i + 1}/{len(images)} 처리 중...")
            result = self.classify(image)
            results.append(result)

        return results

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict:
        """classify 메서드 호출"""
        return self.classify(image)
