"""YOLO 기반 문서 레이아웃 검출 모듈"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ..utils.logger import LoggerMixin


class YOLODetector(LoggerMixin):
    """YOLO 기반 문서 요소 검출기"""

    DETECTION_CLASSES = [
        "stamp",
        "signature",
        "table",
        "barcode",
        "qrcode",
        "hospital_logo"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Args:
            model_path: 학습된 YOLO 모델 경로 (None이면 사전학습 모델 사용)
            confidence_threshold: 검출 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            device: 실행 장치 ("auto", "cpu", "cuda")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """YOLO 모델 로드"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 패키지를 설치해주세요: pip install ultralytics")

        if self.model_path and Path(self.model_path).exists():
            self.logger.info(f"학습된 모델 로드: {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            self.logger.info("사전학습 YOLOv8n 모델 사용")
            self.model = YOLO("yolov8n.pt")

        self.logger.info("YOLO 모델 로드 완료")

    def detect(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        verbose: bool = False
    ) -> Dict:
        """이미지에서 문서 요소 검출

        Args:
            image: 입력 이미지 (numpy array, PIL Image, 또는 경로)
            verbose: 상세 출력 여부

        Returns:
            검출 결과 딕셔너리:
            {
                "detections": [
                    {
                        "class": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy]
                    },
                    ...
                ],
                "image_size": [width, height],
                "detection_count": int
            }
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )

        result = results[0]
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                class_name = result.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "width": x2 - x1,
                    "height": y2 - y1
                })

        orig_shape = result.orig_shape
        image_size = [orig_shape[1], orig_shape[0]]

        self.logger.debug(f"검출 완료: {len(detections)}개 객체")

        return {
            "detections": detections,
            "image_size": image_size,
            "detection_count": len(detections)
        }

    def get_layout_features(self, detection_result: Dict) -> Dict:
        """검출 결과에서 레이아웃 특징 추출

        Args:
            detection_result: detect() 메서드의 반환값

        Returns:
            레이아웃 특징 딕셔너리
        """
        detections = detection_result["detections"]
        image_w, image_h = detection_result["image_size"]
        image_area = image_w * image_h

        features = {
            "has_stamp": False,
            "has_hospital_logo": False,
            "has_barcode": False,
            "has_qrcode": False,
            "has_table": False,
            "has_signature": False,
            "stamp_count": 0,
            "table_count": 0,
            "logo_position": None,
            "table_area_ratio": 0.0,
            "detection_summary": {}
        }

        for det in detections:
            cls = det["class"]
            features["detection_summary"].setdefault(cls, []).append(det["confidence"])

            if cls == "stamp":
                features["has_stamp"] = True
                features["stamp_count"] += 1
            elif cls == "hospital_logo":
                features["has_hospital_logo"] = True
                features["logo_position"] = det["center"]
            elif cls == "barcode":
                features["has_barcode"] = True
            elif cls == "qrcode":
                features["has_qrcode"] = True
            elif cls == "table":
                features["has_table"] = True
                features["table_count"] += 1
                table_area = det["width"] * det["height"]
                features["table_area_ratio"] += table_area / image_area
            elif cls == "signature":
                features["has_signature"] = True

        for cls in features["detection_summary"]:
            confs = features["detection_summary"][cls]
            features["detection_summary"][cls] = {
                "count": len(confs),
                "avg_confidence": sum(confs) / len(confs),
                "max_confidence": max(confs)
            }

        return features

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """검출 결과 시각화

        Args:
            image: 입력 이미지
            save_path: 저장 경로 (선택)

        Returns:
            시각화된 이미지
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        annotated = results[0].plot()

        if save_path:
            import cv2
            cv2.imwrite(save_path, annotated)
            self.logger.info(f"시각화 결과 저장: {save_path}")

        return annotated

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict:
        """detect 메서드 호출"""
        return self.detect(image)
