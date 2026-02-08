#!/usr/bin/env python3
"""YOLO 모델 학습 스크립트"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.step1_yolo.train import YOLOTrainer
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="YOLO 문서 레이아웃 검출 모델 학습")

    parser.add_argument(
        "--data-yaml",
        required=True,
        help="데이터셋 YAML 파일 경로"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="사전학습 모델 (기본: yolov8n.pt)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에폭 수 (기본: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="배치 크기 (기본: 16)"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="입력 이미지 크기 (기본: 640)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="학습률 (기본: 0.01)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="출력 디렉토리 (기본: data/models)"
    )

    parser.add_argument(
        "--project-name",
        default="document_layout",
        help="프로젝트 이름 (기본: document_layout)"
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="학습 장치 (기본: auto)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 학습 이어서 진행"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="검증만 수행"
    )

    parser.add_argument(
        "--export",
        choices=["onnx", "torchscript", "coreml"],
        help="학습 후 모델 내보내기"
    )

    args = parser.parse_args()
    logger = get_logger("train_yolo")

    trainer = YOLOTrainer(
        data_yaml=args.data_yaml,
        model_name=args.model,
        output_dir=args.output_dir,
        project_name=args.project_name
    )

    if args.validate_only:
        logger.info("모델 검증 모드")
        metrics = trainer.validate()
        print("\n검증 결과:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        return

    logger.info("YOLO 학습 시작")
    result = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=args.device,
        resume=args.resume
    )

    print(f"\n학습 완료!")
    print(f"모델 저장 위치: {result['model_path']}")

    logger.info("학습 후 검증")
    metrics = trainer.validate(model_path=result['model_path'])
    print("\n검증 결과:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    if args.export:
        logger.info(f"모델 내보내기: {args.export}")
        export_path = trainer.export(
            model_path=result['model_path'],
            format=args.export
        )
        print(f"\n내보내기 완료: {export_path}")


if __name__ == "__main__":
    main()
