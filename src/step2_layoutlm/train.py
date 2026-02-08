"""LayoutLM 모델 학습 모듈"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ..utils.logger import LoggerMixin


class DocumentDataset(Dataset):
    """문서 분류 데이터셋"""

    def __init__(
        self,
        data_dir: str,
        processor,
        label2id: Dict[str, int],
        max_seq_length: int = 512
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 (images/, annotations/ 포함)
            processor: LayoutLMv3Processor
            label2id: 레이블-ID 매핑
            max_seq_length: 최대 시퀀스 길이
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.label2id = label2id
        self.max_seq_length = max_seq_length

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """데이터 샘플 로드"""
        samples = []

        annotations_dir = self.data_dir / "annotations"
        images_dir = self.data_dir / "images"

        if not annotations_dir.exists():
            raise FileNotFoundError(f"annotations 디렉토리를 찾을 수 없습니다: {annotations_dir}")

        for annotation_file in annotations_dir.glob("*.json"):
            with open(annotation_file, "r", encoding="utf-8") as f:
                annotation = json.load(f)

            image_name = annotation.get("image", annotation_file.stem + ".jpg")
            image_path = images_dir / image_name

            if image_path.exists():
                samples.append({
                    "image_path": str(image_path),
                    "words": annotation.get("words", []),
                    "boxes": annotation.get("boxes", []),
                    "label": annotation.get("label")
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        words = sample["words"] if sample["words"] else ["[EMPTY]"]
        boxes = sample["boxes"] if sample["boxes"] else [[0, 0, 0, 0]]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        if sample["label"] in self.label2id:
            encoding["labels"] = torch.tensor(self.label2id[sample["label"]])
        else:
            encoding["labels"] = torch.tensor(-1)

        return encoding


class LayoutLMTrainer(LoggerMixin):
    """LayoutLM 문서 분류 모델 학습기"""

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
        train_data_dir: str,
        val_data_dir: Optional[str] = None,
        model_name: str = "microsoft/layoutlmv3-base",
        output_dir: str = "data/models/layoutlm_classifier",
        max_seq_length: int = 512
    ):
        """
        Args:
            train_data_dir: 학습 데이터 디렉토리
            val_data_dir: 검증 데이터 디렉토리
            model_name: 사전학습 모델 이름
            output_dir: 출력 디렉토리
            max_seq_length: 최대 시퀀스 길이
        """
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length

        self.id2label = {i: label for i, label in enumerate(self.DOCUMENT_CLASSES)}
        self.label2id = {label: i for i, label in enumerate(self.DOCUMENT_CLASSES)}

        self.model = None
        self.processor = None

    def setup(self) -> None:
        """모델 및 프로세서 초기화"""
        try:
            from transformers import (
                LayoutLMv3ForSequenceClassification,
                LayoutLMv3Processor
            )
        except ImportError:
            raise ImportError(
                "transformers 패키지를 설치해주세요: pip install transformers"
            )

        self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.DOCUMENT_CLASSES),
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.logger.info(f"모델 초기화 완료: {self.model_name}")

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        device: str = "auto",
        save_steps: int = 500,
        eval_steps: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """모델 학습

        Args:
            epochs: 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
            warmup_ratio: 워밍업 비율
            weight_decay: 가중치 감쇠
            device: 학습 장치
            save_steps: 저장 간격
            eval_steps: 평가 간격
            **kwargs: 추가 학습 파라미터

        Returns:
            학습 결과
        """
        from transformers import TrainingArguments, Trainer

        if self.model is None:
            self.setup()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        train_dataset = DocumentDataset(
            self.train_data_dir,
            self.processor,
            self.label2id,
            self.max_seq_length
        )

        eval_dataset = None
        if self.val_data_dir:
            eval_dataset = DocumentDataset(
                self.val_data_dir,
                self.processor,
                self.label2id,
                self.max_seq_length
            )

        self.logger.info(f"학습 데이터: {len(train_dataset)}개")
        if eval_dataset:
            self.logger.info(f"검증 데이터: {len(eval_dataset)}개")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="accuracy" if eval_dataset else None,
            report_to="none",
            **kwargs
        )

        def compute_metrics(eval_pred):
            import numpy as np
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None
        )

        self.logger.info("학습 시작")
        train_result = trainer.train()

        trainer.save_model(str(self.output_dir / "final"))
        self.processor.save_pretrained(str(self.output_dir / "final"))

        self.logger.info(f"모델 저장 완료: {self.output_dir / 'final'}")

        return {
            "model_path": str(self.output_dir / "final"),
            "train_loss": train_result.training_loss,
            "metrics": train_result.metrics
        }

    def evaluate(
        self,
        model_path: Optional[str] = None,
        test_data_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """모델 평가

        Args:
            model_path: 평가할 모델 경로
            test_data_dir: 테스트 데이터 디렉토리

        Returns:
            평가 메트릭
        """
        from transformers import (
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3Processor
        )
        from sklearn.metrics import classification_report, accuracy_score

        model_path = model_path or str(self.output_dir / "final")
        test_data_dir = test_data_dir or self.val_data_dir

        if not test_data_dir:
            raise ValueError("테스트 데이터 디렉토리가 필요합니다")

        model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
        processor = LayoutLMv3Processor.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        test_dataset = DocumentDataset(
            test_data_dir,
            processor,
            self.label2id,
            self.max_seq_length
        )

        dataloader = DataLoader(test_dataset, batch_size=8)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.DOCUMENT_CLASSES,
            output_dict=True
        )

        self.logger.info(f"평가 완료 - 정확도: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "classification_report": report
        }
