"""파이프라인 테스트"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DocumentClassificationPipeline, ClassificationResult
from src.preprocessor import Deskewer, DocumentDetector
from src.step1_yolo import YOLODetector, YOLOClassifier
from src.step2_layoutlm import OCRProcessor, LayoutLMClassifier
from src.step3_vlm import VLMHandler, DocumentClassificationPrompts


class TestDeskewer:
    """기울기 보정 테스트"""

    def test_init(self):
        deskewer = Deskewer(max_angle=10)
        assert deskewer.max_angle == 10

    def test_deskew_no_rotation_needed(self):
        deskewer = Deskewer()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result, angle = deskewer.deskew(image)

        assert angle == 0.0
        assert result.shape[:2] == image.shape[:2]

    def test_deskew_with_angle(self):
        deskewer = Deskewer()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result, angle = deskewer.deskew(image, angle=5.0)

        assert angle == 5.0


class TestDocumentDetector:
    """문서 영역 검출 테스트"""

    def test_init(self):
        detector = DocumentDetector(min_area_ratio=0.5)
        assert detector.min_area_ratio == 0.5

    def test_detect_no_document(self):
        detector = DocumentDetector()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result, contour = detector.detect_and_extract(image)

        assert contour is None

    def test_order_points(self):
        detector = DocumentDetector()
        pts = np.array([
            [100, 0],
            [0, 0],
            [0, 100],
            [100, 100]
        ])

        ordered = detector.order_points(pts)

        assert ordered[0].tolist() == [0, 0]
        assert ordered[1].tolist() == [100, 0]
        assert ordered[2].tolist() == [100, 100]
        assert ordered[3].tolist() == [0, 100]


class TestYOLODetector:
    """YOLO 검출기 테스트"""

    def test_init(self):
        detector = YOLODetector(confidence_threshold=0.6)
        assert detector.confidence_threshold == 0.6
        assert detector.model is None

    @patch("src.step1_yolo.detector.YOLO")
    def test_load_model(self, mock_yolo):
        detector = YOLODetector()
        detector.load_model()

        mock_yolo.assert_called_once()
        assert detector.model is not None

    def test_get_layout_features(self):
        detector = YOLODetector()

        detection_result = {
            "detections": [
                {"class": "stamp", "confidence": 0.9, "width": 50, "height": 50},
                {"class": "table", "confidence": 0.8, "width": 200, "height": 300},
            ],
            "image_size": [640, 480],
            "detection_count": 2
        }

        features = detector.get_layout_features(detection_result)

        assert features["has_stamp"] is True
        assert features["has_table"] is True
        assert features["stamp_count"] == 1
        assert features["table_count"] == 1


class TestYOLOClassifier:
    """YOLO 분류기 테스트"""

    def test_init(self):
        classifier = YOLOClassifier()
        assert classifier.detector is not None

    def test_calculate_scores(self):
        classifier = YOLOClassifier()

        features = {
            "has_stamp": True,
            "has_hospital_logo": True,
            "has_table": False,
            "has_barcode": False,
            "has_signature": True,
            "table_area_ratio": 0.1
        }

        scores = classifier._calculate_scores(features)

        assert "진단서" in scores
        assert "소견서" in scores
        assert all(0 <= score <= 1 for score in scores.values())


class TestOCRProcessor:
    """OCR 처리기 테스트"""

    def test_init(self):
        processor = OCRProcessor(engine="paddleocr")
        assert processor.engine == "paddleocr"
        assert processor.ocr is None

    def test_normalize_boxes(self):
        processor = OCRProcessor()

        boxes = [[0, 0, 100, 50], [50, 50, 150, 100]]
        image_size = (200, 100)

        normalized = processor.normalize_boxes(boxes, image_size)

        assert normalized[0] == [0, 0, 500, 500]
        assert normalized[1] == [250, 500, 750, 1000]


class TestLayoutLMClassifier:
    """LayoutLM 분류기 테스트"""

    def test_init(self):
        classifier = LayoutLMClassifier(confidence_threshold=0.8)
        assert classifier.confidence_threshold == 0.8
        assert classifier.model is None

    def test_get_top_predictions(self):
        classifier = LayoutLMClassifier()

        result = {
            "all_probabilities": {
                "진단서": 0.8,
                "소견서": 0.1,
                "보험금청구서": 0.05,
                "입퇴원확인서": 0.03,
                "의료비영수증": 0.01,
                "처방전": 0.01
            }
        }

        top_3 = classifier.get_top_predictions(result, top_k=3)

        assert len(top_3) == 3
        assert top_3[0][0] == "진단서"
        assert top_3[0][1] == 0.8


class TestDocumentClassificationPrompts:
    """VLM 프롬프트 테스트"""

    def test_get_classification_prompt_basic(self):
        prompt = DocumentClassificationPrompts.get_classification_prompt()
        assert "분류" in prompt
        assert "진단서" in prompt

    def test_get_classification_prompt_with_context(self):
        step1_result = {"predicted_class": "진단서", "confidence": 0.7}
        step2_result = {"predicted_class": "소견서", "confidence": 0.6}

        prompt = DocumentClassificationPrompts.get_classification_prompt(
            step1_result, step2_result
        )

        assert "진단서" in prompt
        assert "소견서" in prompt
        assert "0.70" in prompt or "0.7" in prompt

    def test_parse_response(self):
        response = """분류: 진단서
신뢰도: 높음
근거: 문서에 진단명과 의사 서명이 포함되어 있습니다."""

        result = DocumentClassificationPrompts.parse_response(response)

        assert result["predicted_class"] == "진단서"
        assert result["confidence_level"] == "high"
        assert "진단명" in result["reasoning"]


class TestVLMHandler:
    """VLM 핸들러 테스트"""

    def test_init_openai(self):
        handler = VLMHandler(provider="openai")
        assert handler.provider == "openai"
        assert handler.model == "gpt-4o"

    def test_init_anthropic(self):
        handler = VLMHandler(provider="anthropic")
        assert handler.provider == "anthropic"
        assert "claude" in handler.model

    def test_invalid_provider(self):
        with pytest.raises(ValueError):
            VLMHandler(provider="invalid")


class TestClassificationResult:
    """분류 결과 테스트"""

    def test_to_dict(self):
        result = ClassificationResult(
            predicted_class="진단서",
            confidence=0.85,
            final_step=2,
            step1_result={"predicted_class": "진단서"},
            step2_result={"predicted_class": "진단서"}
        )

        result_dict = result.to_dict()

        assert result_dict["predicted_class"] == "진단서"
        assert result_dict["confidence"] == 0.85
        assert result_dict["final_step"] == 2


class TestDocumentClassificationPipeline:
    """파이프라인 통합 테스트"""

    def test_init_default(self):
        with patch.object(DocumentClassificationPipeline, '_init_components'):
            pipeline = DocumentClassificationPipeline()

        assert pipeline.enable_preprocessing is True
        assert pipeline.enable_step1 is True
        assert pipeline.enable_step2 is True
        assert pipeline.enable_step3 is True

    def test_init_disabled_steps(self):
        with patch.object(DocumentClassificationPipeline, '_init_components'):
            pipeline = DocumentClassificationPipeline(
                enable_step1=False,
                enable_step3=False
            )

        assert pipeline.enable_step1 is False
        assert pipeline.enable_step2 is True
        assert pipeline.enable_step3 is False

    def test_document_classes(self):
        expected_classes = [
            "진단서",
            "소견서",
            "보험금청구서",
            "입퇴원확인서",
            "의료비영수증",
            "처방전"
        ]

        assert DocumentClassificationPipeline.DOCUMENT_CLASSES == expected_classes

    def test_get_statistics(self):
        with patch.object(DocumentClassificationPipeline, '_init_components'):
            pipeline = DocumentClassificationPipeline()

        results = [
            ClassificationResult("진단서", 0.9, 1, processing_time=0.5),
            ClassificationResult("진단서", 0.85, 2, processing_time=1.0),
            ClassificationResult("소견서", 0.95, 1, processing_time=0.4),
            ClassificationResult("처방전", 0.7, 3, processing_time=2.0),
        ]

        stats = pipeline.get_statistics(results)

        assert stats["total_documents"] == 4
        assert stats["class_distribution"]["진단서"] == 2
        assert stats["class_distribution"]["소견서"] == 1
        assert stats["class_distribution"]["처방전"] == 1
        assert stats["step_distribution"]["step1"] == 2
        assert stats["step_distribution"]["step2"] == 1
        assert stats["step_distribution"]["step3"] == 1
        assert stats["vlm_usage_rate"] == 0.25


class TestIntegration:
    """통합 테스트 (모킹 사용)"""

    @patch("src.step1_yolo.detector.YOLO")
    @patch("src.step2_layoutlm.ocr.PaddleOCR")
    def test_pipeline_step1_only(self, mock_paddleocr, mock_yolo):
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.orig_shape = (480, 640)
        mock_result.names = {}
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        pipeline = DocumentClassificationPipeline(
            enable_step2=False,
            enable_step3=False,
            enable_preprocessing=False
        )

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(pipeline.yolo_classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                "predicted_class": "진단서",
                "confidence": 0.9,
                "requires_step2": False,
                "layout_features": {},
                "detection_result": {}
            }

            result = pipeline.classify(image)

        assert result.predicted_class == "진단서"
        assert result.final_step == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
