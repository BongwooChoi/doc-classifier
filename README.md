# 문서분류기

3단계 파이프라인 기반 의료/보험 문서 분류 시스템

## 개요

이 시스템은 3단계 파이프라인을 통해 의료 및 보험 관련 문서를 자동으로 분류합니다:

- **Step 1 (YOLO)**: 레이아웃 검출 및 1차 분류
- **Step 2 (LayoutLM)**: 텍스트 기반 정밀 분류
- **Step 3 (VLM)**: 예외 처리 (1~5% 문서만)

## 분류 대상 문서

- 진단서
- 소견서
- 보험금청구서
- 입퇴원확인서
- 의료비영수증
- 처방전

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd 문서분류기

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```bash
# 단일 이미지 분류
python main.py /path/to/document.jpg

# 디렉토리 내 모든 이미지 분류
python main.py /path/to/documents/

# 결과를 JSON으로 저장
python main.py /path/to/documents/ -o results.json
```

### 옵션

```bash
python main.py --help

옵션:
  -c, --config        설정 파일 경로 (기본: config/config.yaml)
  -o, --output        결과 출력 파일 (JSON)
  --skip-step1        Step 1 (YOLO) 건너뛰기
  --skip-step2        Step 2 (LayoutLM) 건너뛰기
  --skip-step3        Step 3 (VLM) 건너뛰기
  --skip-preprocessing 전처리 건너뛰기
  --force-all-steps   모든 단계 강제 실행
  -v, --verbose       상세 출력
```

### Python API 사용

```python
from src.pipeline import DocumentClassificationPipeline

# 파이프라인 초기화
pipeline = DocumentClassificationPipeline()

# 이미지 분류
result = pipeline.classify("document.jpg")

print(f"분류 결과: {result.predicted_class}")
print(f"신뢰도: {result.confidence:.2%}")
print(f"최종 단계: Step {result.final_step}")
```

## 학습

### YOLO 모델 학습

```bash
# 데이터 YAML 생성
python scripts/prepare_data.py yolo-yaml data/yolo_dataset

# 모델 학습
python scripts/train_yolo.py \
    --data-yaml data/annotations/yolo_data.yaml \
    --epochs 100 \
    --batch-size 16
```

### LayoutLM 모델 학습

```bash
# 어노테이션 생성
python scripts/prepare_data.py layoutlm-anno \
    data/images data/ocr data/labels.tsv

# 모델 학습
python scripts/train_layoutlm.py \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 10
```

## 평가

```bash
python scripts/evaluate.py \
    --test-dir data/test \
    --labels data/test_labels.tsv \
    --output evaluation_results.json
```

## 설정

`config/config.yaml` 파일에서 설정을 변경할 수 있습니다:

```yaml
# YOLO 설정
yolo:
  model_path: "data/models/yolo_document.pt"
  confidence_threshold: 0.5

# LayoutLM 설정
layoutlm:
  model_name: "microsoft/layoutlmv3-base"
  confidence_threshold: 0.7  # 이 값 미만이면 VLM으로 전달

# VLM 설정
vlm:
  provider: "openai"  # openai 또는 anthropic
  openai:
    model: "gpt-4o"
```

## 프로젝트 구조

```
문서분류기/
├── config/
│   └── config.yaml              # 전체 설정
├── src/
│   ├── pipeline.py              # 메인 파이프라인
│   ├── preprocessor/            # 전처리 모듈
│   ├── step1_yolo/              # YOLO 검출/분류
│   ├── step2_layoutlm/          # LayoutLM 분류
│   ├── step3_vlm/               # VLM 예외 처리
│   └── utils/                   # 유틸리티
├── data/
│   ├── raw/                     # 원본 데이터
│   ├── processed/               # 전처리된 데이터
│   ├── annotations/             # 라벨링 데이터
│   └── models/                  # 학습된 모델
├── scripts/                     # 학습/평가 스크립트
├── tests/                       # 테스트
├── main.py                      # 실행 진입점
└── requirements.txt
```

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ -v --cov=src
```

## 환경 변수

VLM 사용 시 API 키 설정이 필요합니다:

```bash
# OpenAI 사용 시
export OPENAI_API_KEY="your-api-key"

# Anthropic 사용 시
export ANTHROPIC_API_KEY="your-api-key"
```

## 라이선스

MIT License
