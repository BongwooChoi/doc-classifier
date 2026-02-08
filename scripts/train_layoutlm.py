#!/usr/bin/env python3
"""LayoutLM 모델 학습 스크립트"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.step2_layoutlm.train import LayoutLMTrainer
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="LayoutLM 문서 분류 모델 학습")

    parser.add_argument(
        "--train-dir",
        required=True,
        help="학습 데이터 디렉토리"
    )

    parser.add_argument(
        "--val-dir",
        help="검증 데이터 디렉토리"
    )

    parser.add_argument(
        "--model-name",
        default="microsoft/layoutlmv3-base",
        help="사전학습 모델 이름 (기본: microsoft/layoutlmv3-base)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 에폭 수 (기본: 10)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="배치 크기 (기본: 8)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="학습률 (기본: 2e-5)"
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="워밍업 비율 (기본: 0.1)"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="최대 시퀀스 길이 (기본: 512)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/models/layoutlm_classifier",
        help="출력 디렉토리"
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="학습 장치 (기본: auto)"
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="평가만 수행"
    )

    parser.add_argument(
        "--model-path",
        help="평가할 모델 경로 (--evaluate-only 사용 시)"
    )

    args = parser.parse_args()
    logger = get_logger("train_layoutlm")

    trainer = LayoutLMTrainer(
        train_data_dir=args.train_dir,
        val_data_dir=args.val_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length
    )

    if args.evaluate_only:
        if not args.model_path:
            logger.error("--model-path가 필요합니다")
            return

        logger.info("모델 평가 모드")
        metrics = trainer.evaluate(
            model_path=args.model_path,
            test_data_dir=args.val_dir or args.train_dir
        )

        print("\n평가 결과:")
        print(f"  정확도: {metrics['accuracy']:.4f}")
        print("\n클래스별 성능:")
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall: {class_metrics['recall']:.4f}")
                print(f"    F1-Score: {class_metrics['f1-score']:.4f}")
        return

    logger.info("LayoutLM 학습 시작")
    result = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        device=args.device
    )

    print(f"\n학습 완료!")
    print(f"모델 저장 위치: {result['model_path']}")
    print(f"학습 손실: {result['train_loss']:.4f}")

    if args.val_dir:
        logger.info("학습 후 평가")
        metrics = trainer.evaluate(
            model_path=result['model_path'],
            test_data_dir=args.val_dir
        )
        print(f"\n평가 정확도: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
