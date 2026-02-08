#!/usr/bin/env python3
"""문서분류기 실행 진입점"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.pipeline import DocumentClassificationPipeline, ClassificationResult
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="3단계 파이프라인 기반 의료/보험 문서 분류 시스템"
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="입력 이미지 경로 또는 디렉토리"
    )

    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="설정 파일 경로 (기본: config/config.yaml)"
    )

    parser.add_argument(
        "-o", "--output",
        help="결과 출력 파일 (JSON)"
    )

    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Step 1 (YOLO) 건너뛰기"
    )

    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Step 2 (LayoutLM) 건너뛰기"
    )

    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Step 3 (VLM) 건너뛰기"
    )

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="전처리 건너뛰기"
    )

    parser.add_argument(
        "--force-all-steps",
        action="store_true",
        help="모든 단계 강제 실행"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="문서분류기 v0.1.0"
    )

    return parser.parse_args()


def get_image_files(path: Path) -> List[Path]:
    """디렉토리에서 이미지 파일 목록 가져오기"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if path.is_file():
        return [path]

    if path.is_dir():
        files = []
        for ext in image_extensions:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(files)

    return []


def print_result(result: ClassificationResult, image_path: Optional[str] = None) -> None:
    """분류 결과 출력"""
    if image_path:
        print(f"\n{'=' * 60}")
        print(f"파일: {image_path}")

    print(f"분류 결과: {result.predicted_class}")
    print(f"신뢰도: {result.confidence:.2%}")
    print(f"최종 단계: Step {result.final_step}")

    if result.processing_time:
        print(f"처리 시간: {result.processing_time:.2f}초")

    if result.step1_result:
        print(f"\nStep 1 결과: {result.step1_result['predicted_class']} "
              f"(신뢰도: {result.step1_result['confidence']:.2%})")

    if result.step2_result:
        print(f"Step 2 결과: {result.step2_result['predicted_class']} "
              f"(신뢰도: {result.step2_result['confidence']:.2%})")

    if result.step3_result:
        print(f"Step 3 결과: {result.step3_result['predicted_class']} "
              f"(근거: {result.step3_result.get('reasoning', 'N/A')})")


def main() -> int:
    """메인 함수"""
    args = parse_args()
    logger = get_logger("main", level="DEBUG" if args.verbose else "INFO")

    if not args.input:
        print("사용법: python main.py <이미지 경로 또는 디렉토리>")
        print("도움말: python main.py --help")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 경로를 찾을 수 없습니다: {input_path}")
        return 1

    image_files = get_image_files(input_path)
    if not image_files:
        logger.error(f"처리할 이미지 파일이 없습니다: {input_path}")
        return 1

    logger.info(f"처리할 이미지: {len(image_files)}개")

    try:
        pipeline = DocumentClassificationPipeline(
            config_path=args.config,
            enable_preprocessing=not args.skip_preprocessing,
            enable_step1=not args.skip_step1,
            enable_step2=not args.skip_step2,
            enable_step3=not args.skip_step3
        )
    except Exception as e:
        logger.error(f"파이프라인 초기화 실패: {e}")
        return 1

    results = []

    for image_path in image_files:
        try:
            logger.info(f"처리 중: {image_path.name}")

            result = pipeline.classify(
                str(image_path),
                force_all_steps=args.force_all_steps
            )

            results.append({
                "file": str(image_path),
                "result": result.to_dict()
            })

            print_result(result, str(image_path))

        except Exception as e:
            logger.error(f"처리 실패 ({image_path.name}): {e}")
            results.append({
                "file": str(image_path),
                "error": str(e)
            })

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"결과 저장: {output_path}")

    if len(image_files) > 1:
        print(f"\n{'=' * 60}")
        print(f"총 처리: {len(image_files)}개 문서")

        from collections import Counter
        class_counts = Counter(
            r["result"]["predicted_class"]
            for r in results
            if "result" in r
        )
        print("\n분류 결과 분포:")
        for cls, count in class_counts.most_common():
            print(f"  - {cls}: {count}개")

    return 0


if __name__ == "__main__":
    sys.exit(main())
