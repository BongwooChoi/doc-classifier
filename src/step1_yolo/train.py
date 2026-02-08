"""YOLO 모델 학습 모듈"""

from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.logger import LoggerMixin


class YOLOTrainer(LoggerMixin):
    """YOLO 문서 레이아웃 검출 모델 학습기"""

    def __init__(
        self,
        data_yaml: str,
        model_name: str = "yolov8n.pt",
        output_dir: str = "data/models",
        project_name: str = "document_layout"
    ):
        """
        Args:
            data_yaml: 데이터셋 설정 YAML 파일 경로
            model_name: 사전학습 모델 이름
            output_dir: 출력 디렉토리
            project_name: 프로젝트 이름
        """
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.model = None

    def prepare_data_yaml(
        self,
        train_path: str,
        val_path: str,
        classes: list,
        output_path: str
    ) -> str:
        """YOLO 데이터셋 YAML 파일 생성

        Args:
            train_path: 학습 이미지 경로
            val_path: 검증 이미지 경로
            classes: 클래스 목록
            output_path: YAML 파일 저장 경로

        Returns:
            생성된 YAML 파일 경로
        """
        import yaml

        data_config = {
            "path": str(Path(train_path).parent.parent),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)}
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

        self.logger.info(f"데이터셋 YAML 생성: {output_path}")
        return str(output_path)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        learning_rate: float = 0.01,
        device: str = "auto",
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """모델 학습

        Args:
            epochs: 에폭 수
            batch_size: 배치 크기
            image_size: 입력 이미지 크기
            learning_rate: 학습률
            device: 학습 장치
            resume: 이전 학습 이어서 진행 여부
            **kwargs: 추가 학습 파라미터

        Returns:
            학습 결과 딕셔너리
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 패키지를 설치해주세요: pip install ultralytics")

        if not Path(self.data_yaml).exists():
            raise FileNotFoundError(f"데이터셋 YAML 파일을 찾을 수 없습니다: {self.data_yaml}")

        self.logger.info(f"YOLO 학습 시작: {self.model_name}")
        self.logger.info(f"- 에폭: {epochs}")
        self.logger.info(f"- 배치 크기: {batch_size}")
        self.logger.info(f"- 이미지 크기: {image_size}")

        self.model = YOLO(self.model_name)

        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            lr0=learning_rate,
            device=device,
            project=str(self.output_dir),
            name=self.project_name,
            resume=resume,
            exist_ok=True,
            **kwargs
        )

        self.logger.info("YOLO 학습 완료")

        return {
            "model_path": str(self.output_dir / self.project_name / "weights" / "best.pt"),
            "results": results
        }

    def validate(
        self,
        model_path: Optional[str] = None,
        data_yaml: Optional[str] = None
    ) -> Dict[str, Any]:
        """모델 검증

        Args:
            model_path: 검증할 모델 경로
            data_yaml: 데이터셋 YAML 경로

        Returns:
            검증 결과
        """
        from ultralytics import YOLO

        model_path = model_path or str(
            self.output_dir / self.project_name / "weights" / "best.pt"
        )
        data_yaml = data_yaml or self.data_yaml

        self.logger.info(f"모델 검증: {model_path}")

        model = YOLO(model_path)
        results = model.val(data=data_yaml)

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr)
        }

        self.logger.info(f"검증 결과: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}")

        return metrics

    def export(
        self,
        model_path: Optional[str] = None,
        format: str = "onnx"
    ) -> str:
        """모델 내보내기

        Args:
            model_path: 내보낼 모델 경로
            format: 내보내기 형식 (onnx, torchscript, etc.)

        Returns:
            내보낸 모델 경로
        """
        from ultralytics import YOLO

        model_path = model_path or str(
            self.output_dir / self.project_name / "weights" / "best.pt"
        )

        self.logger.info(f"모델 내보내기: {format}")

        model = YOLO(model_path)
        export_path = model.export(format=format)

        self.logger.info(f"내보내기 완료: {export_path}")

        return export_path
