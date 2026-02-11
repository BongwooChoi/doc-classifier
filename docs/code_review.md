# 📋 문서분류기 프로젝트 코드 리뷰

**리뷰 일자**: 2026-02-09  
**프로젝트 버전**: v0.1.0  
**리뷰 범위**: 전체 소스 코드, 아키텍처, 테스트, 데이터, 배포

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처 리뷰](#2-아키텍처-리뷰)
3. [모듈별 상세 리뷰](#3-모듈별-상세-리뷰)
4. [데이터 파이프라인 리뷰](#4-데이터-파이프라인-리뷰)
5. [API 서버 리뷰](#5-api-서버-리뷰)
6. [테스트 커버리지 리뷰](#6-테스트-커버리지-리뷰)
7. [코드 품질 분석](#7-코드-품질-분석)
8. [보안 및 에러 처리 리뷰](#8-보안-및-에러-처리-리뷰)
9. [성능 분석](#9-성능-분석)
10. [현재 결과 분석 (results.json)](#10-현재-결과-분석)
11. [종합 평가 및 개선 로드맵](#11-종합-평가-및-개선-로드맵)

---

## 1. 프로젝트 개요

### 1.1 시스템 목적
의료/보험 문서 6종(진단서, 소견서, 보험금청구서, 입퇴원확인서, 의료비영수증, 처방전)을 자동 분류하는 3단계 파이프라인 시스템.

### 1.2 기술 스택 요약

| 구성 요소 | 기술 | 버전 |
|-----------|------|------|
| Step 1 검출/분류 | YOLOv8n (ultralytics) | >= 8.0.0 |
| Step 2 정밀 분류 | LayoutLMv3 (HuggingFace) | transformers >= 4.30.0 |
| Step 3 예외 처리 | GPT-4o / Claude 3.5 Sonnet | openai >= 1.0.0, anthropic >= 0.18.0 |
| OCR 엔진 | EasyOCR (기본) / PaddleOCR | easyocr >= 1.7.0 |
| 전처리 | OpenCV | >= 4.8.0 |
| API 서버 | FastAPI + Uvicorn | - |
| 언어 | Python 3.12 | - |

### 1.3 디렉토리 구조

```
문서분류기/
├── api/app.py                    # FastAPI REST API
├── config/config.yaml            # 전체 설정 관리
├── src/
│   ├── pipeline.py               # 메인 파이프라인 오케스트레이터
│   ├── preprocessor/
│   │   ├── deskew.py             # 기울기 보정 (Hough Transform)
│   │   └── document_detector.py  # 문서 영역 검출 (Contour Detection)
│   ├── step1_yolo/
│   │   ├── detector.py           # YOLO 객체 검출
│   │   ├── classifier.py         # 레이아웃 기반 규칙 분류
│   │   └── train.py              # YOLO 학습기
│   ├── step2_layoutlm/
│   │   ├── classifier.py         # LayoutLMv3 분류기
│   │   ├── ocr.py                # OCR 처리 (PaddleOCR/EasyOCR)
│   │   └── train.py              # LayoutLM 학습기
│   ├── step3_vlm/
│   │   ├── handler.py            # VLM API 호출 핸들러
│   │   └── prompts.py            # 프롬프트 템플릿 관리
│   └── utils/
│       ├── logger.py             # 로깅 유틸리티
│       └── image_utils.py        # 이미지 처리 유틸리티
├── scripts/                      # 학습/테스트/평가 스크립트 (15개)
├── tests/test_pipeline.py        # 유닛/통합 테스트
├── data/
│   ├── sample/                   # 합성 데이터 (train 48장, test 12장)
│   ├── yolo_dataset/             # YOLO 학습 데이터
│   └── models/                   # 학습된 모델 가중치
├── main.py                       # CLI 진입점
├── run_api.py                    # API 서버 실행
└── requirements.txt              # 의존성 패키지
```

---

## 2. 아키텍처 리뷰

### 2.1 파이프라인 흐름도

```
입력 이미지
    │
    ▼
[전처리] ─── Deskewer (기울기 보정) ──→ DocumentDetector (문서 영역 검출)
    │
    ▼
[Step 1: YOLO] ─── YOLODetector (요소 검출) ──→ YOLOClassifier (규칙 기반 분류)
    │                                               │
    │  신뢰도 >= threshold ──────────────────────→ 결과 반환
    │  신뢰도 < threshold
    ▼
[Step 2: LayoutLM] ─── OCRProcessor (텍스트 추출) ──→ LayoutLMClassifier (정밀 분류)
    │                                                   │
    │  신뢰도 >= threshold ──────────────────────────→ 결과 반환
    │  신뢰도 < threshold
    ▼
[Step 3: VLM] ─── VLMHandler (GPT-4o/Claude) ──→ 최종 결과 반환
```

### 2.2 아키텍처 강점 ✅

| 항목 | 설명 |
|------|------|
| **단계적 에스컬레이션** | 비용/속도 효율적인 계단식 분류 (YOLO→LayoutLM→VLM). 대부분의 문서는 Step 1-2에서 처리되므로 고비용 VLM 호출 최소화 |
| **모듈화** | 각 Step이 독립 모듈로 분리되어 개별 교체·확장 가능 |
| **설정 기반 구성** | `config.yaml`로 모든 파라미터를 외부에서 제어 |
| **다양한 인터페이스** | CLI (`main.py`), Python API (`pipeline.classify()`), REST API (`api/app.py`) 3종 제공 |
| **Lazy Loading** | 모델이 실제 필요 시점에 로드되어 초기 메모리 절약 |
| **유연한 Step 제어** | `enable_step1/2/3`, `skip_step1/2/3`, `force_all_steps` 등 세밀한 실행 제어 가능 |

### 2.3 아키텍처 개선점 ⚠️

| 항목 | 현재 상태 | 개선 방향 | 심각도 |
|------|----------|----------|--------|
| **Step 간 데이터 전달** | Step 1 결과가 Step 2로 전달되지만, Step 2의 `classify()`가 `step1_result`를 직접 활용하지 않음 (신뢰도 보정만) | Step 1의 레이아웃 특징을 Step 2 입력 feature로 결합하여 앙상블 효과 도출 | 중 |
| **에러 전파** | Step 1 실패 시 전체 파이프라인이 중단됨 | 각 Step 실패 시 다음 Step으로 자동 fallback하는 회복 로직 필요 | 상 |
| **캐싱 부재** | 동일 이미지에 대해 OCR/검출 결과 재계산 | LRU 캐시로 중복 처리 방지 | 중 |
| **비동기 처리 부재** | `batch_classify`가 순차 처리 | `asyncio` 또는 멀티프로세싱으로 배치 병렬화 | 중 |
| **모델 버전 관리** | 모델 파일이 로컬 경로로만 관리됨 | MLflow 등 모델 레지스트리 도입 고려 | 하 |

---

## 3. 모듈별 상세 리뷰

### 3.1 `src/pipeline.py` — 메인 파이프라인

**파일 크기**: 368줄 | **클래스 수**: 2개 (`ClassificationResult`, `DocumentClassificationPipeline`)

#### 잘된 점 ✅
- `ClassificationResult`를 `@dataclass`로 구현하여 타입 안전성과 가독성 확보
- `to_dict()` 메서드 제공으로 JSON 직렬화 용이
- `get_statistics()` 메서드로 결과 분석 기능 내장
- `__call__` 매직 메서드로 호출 편의성 제공

#### 리뷰 이슈 🔍

**[P1] `classify()` 메서드의 시간 측정 방식**
```python
# pipeline.py:212
import time  # 함수 내부에서 import — 매 호출마다 import 오버헤드
start_time = time.time()
```
→ `time` import를 파일 상단으로 이동하고, `time.perf_counter()`로 변경 권장 (더 정밀한 측정)

**[P2] `DOCUMENT_CLASSES` 중복 정의**
```python
# pipeline.py:55, classifier.py(step1):16, classifier.py(step2):17, prompts.py:9, train.py:99, app.py:23
DOCUMENT_CLASSES = ["진단서", "소견서", "보험금청구서", ...]
```
→ **6곳에서 동일한 리스트가 독립적으로 정의**됨. 하나의 중앙 상수로 통합 필요. 문서 유형 추가/변경 시 모든 곳을 수동 수정해야 하므로 버그 발생 위험.

**[P3] `enable_preprocessing=False` 시에도 Deskewer/DocumentDetector 인스턴스화**
```python
# pipeline.py:117-128
if self.enable_preprocessing:
    # ... Deskewer, DocumentDetector 생성
```
→ 이 부분은 올바르게 되어 있음. 다만, `preprocess()` 메서드에서 `self.deskewer`/`self.document_detector` None 체크는 있으나, `enable_preprocessing` 플래그 체크가 `classify()` 내에서만 이루어져 혼란 가능.

**[P4] `batch_classify()` 에러 처리 없음**
```python
# pipeline.py:318
for image in images:
    result = self.classify(image)  # 하나라도 실패하면 전체 중단
    results.append(result)
```
→ `main.py`에서는 개별 try-except 처리가 있지만, `batch_classify()`는 에러 무시/수집 옵션이 없음.

---

### 3.2 `src/step1_yolo/detector.py` — YOLO 검출기

**파일 크기**: 230줄 | **클래스**: `YOLODetector`

#### 잘된 점 ✅
- 검출 결과를 구조화된 딕셔너리로 반환
- `get_layout_features()`로 고수준 특징 추출 분리
- 시각화 기능 (`visualize()`) 내장

#### 리뷰 이슈 🔍

**[P1] 사전학습 모델 fallback 로직 문제**
```python
# detector.py:51-56
if self.model_path and Path(self.model_path).exists():
    self.model = YOLO(self.model_path)
else:
    self.model = YOLO("yolov8n.pt")  # COCO 사전학습 모델
```
→ 학습된 모델이 없으면 COCO 모델로 fallback하는데, COCO 모델의 클래스(person, car 등)와 문서 요소 클래스(stamp, table 등)가 완전히 다름. 의미 있는 검출이 불가능한 상태에서 파이프라인이 정상 작동하는 것처럼 보이는 **Silent Failure** 문제.

**[P2] `detect()` 메서드의 타입 안전성**
```python
# detector.py:110
class_name = result.names.get(cls_id, f"class_{cls_id}")
```
→ 커스텀 모델의 클래스 이름이 `DETECTION_CLASSES`와 매칭되지 않을 수 있음. 검증 로직 부재.

---

### 3.3 `src/step1_yolo/classifier.py` — YOLO 규칙 기반 분류기

**파일 크기**: 240줄 | **클래스**: `YOLOClassifier`

#### 잘된 점 ✅
- 문서 유형별 분류 규칙(`CLASSIFICATION_RULES`)이 선언적으로 정의됨
- 필수/선택/금지 요소, 테이블 비율 등 다층적 점수 계산
- `get_top_predictions()` 유틸리티 메서드 제공

#### 리뷰 이슈 🔍

**[P1-Critical] 진단서/소견서 분류 불가 — 설계적 한계**
```python
# classifier.py:26-39
"진단서": {
    "required": ["stamp", "hospital_logo"],
    "forbidden": ["table", "barcode", "qrcode"],
},
"소견서": {
    "required": ["stamp", "hospital_logo"],
    "forbidden": ["table", "barcode", "qrcode"],
},
```
→ 진단서와 소견서의 규칙이 거의 동일하여 레이아웃만으로는 구분 불가. **이는 설계 의도대로** Step 2에서 텍스트 기반으로 해결하는 것이지만, `results.json`에서 보듯 실제로 Step 2로 넘어가지 않고 Step 1에서 잘못된 결과를 반환하는 문제가 있음 (아래 Section 10 참조).

**[P2] 점수 계산 알고리즘의 엣지 케이스**
```python
# classifier.py:186
score += 0.2 * (1 - (min_ratio - table_ratio) / max(min_ratio, 0.01))
```
→ `table_ratio`가 0이고 `min_ratio`가 0이면 0으로 나누기 방지는 되어 있으나, 결과값이 비직관적. 테이블이 없는 문서에서 진단서 `table_ratio_range: (0.0, 0.1)`이므로 항상 0.4 점수를 받아 유리해짐.

**[P3] 필수 요소 하나도 없을 때 penalty 설정**
```python
# classifier.py:164
if len(rules["required"]) > 0 and required_count == 0:
    penalty += 0.5
```
→ 현재 `results.json`에서 보이는 것처럼, YOLO가 아무 객체도 검출하지 못하면 모든 문서 유형에 동일한 penalty가 적용되고, 테이블 비율과 weight 차이에 의해 "진단서"가 항상 최고 점수를 받는 구조. **모든 테스트 이미지에서 "진단서"만 반환되는 근본 원인**.

---

### 3.4 `src/step2_layoutlm/classifier.py` — LayoutLM 분류기

**파일 크기**: 272줄 | **클래스**: `LayoutLMClassifier`

#### 잘된 점 ✅
- HuggingFace Transformers 표준 패턴 준수 (from_pretrained, apply_ocr=False)
- Step 1 결과와의 일치 시 신뢰도 보정 (`confidence * 1.1`) — 간단하지만 효과적
- 자동 device 선택 (CUDA/CPU)

#### 리뷰 이슈 🔍

**[P1] `batch_classify()`가 실제 배치 처리가 아님**
```python
# classifier.py:253-261
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    for image in batch:  # 결국 하나씩 처리
        result = self.classify(image)
```
→ 배치로 묶긴 하지만 실제 GPU 배치 추론은 하지 않음. `self.processor()`와 `self.model()`에 배치 입력을 전달해야 실제 배치 효과.

**[P2] OCR 텍스트 비어있을 때 처리**
```python
# classifier.py:134-136
if not words:
    words = ["[EMPTY]"]
    boxes = [[0, 0, 0, 0]]
```
→ 빈 문서에 대해 `[EMPTY]` 토큰으로 대체하는 것은 합리적이지만, 이 경우 분류 결과의 신뢰도가 낮아야 함. 현재는 모델이 `[EMPTY]`에 대해서도 높은 확률을 줄 수 있음.

**[P3] `__call__` 시그니처 불일치**
```python
# classifier.py:265-271
def __call__(self, image, ocr_result=None) -> Dict:
    return self.classify(image, ocr_result)

# pipeline.py:251-254 에서 호출
step2_result = self.layoutlm_classifier(image, step1_result=step1_result)
```
→ `__call__`은 `ocr_result`만 받지만, `pipeline.py`에서는 `step1_result`로 호출. **키워드 인자이므로 실제 오류는 나지 않지만**, `step1_result`가 `ocr_result` 파라미터에 매핑되어 의도하지 않은 동작 가능. 실제로 `classify()` 메서드의 시그니처를 보면 `(image, ocr_result, step1_result)` 순서이므로, `pipeline.py`에서 키워드로 정확히 넘겨서 문제는 없지만 `__call__`에서는 `step1_result`를 받지 않는 불일치 존재.

---

### 3.5 `src/step2_layoutlm/ocr.py` — OCR 처리기

**파일 크기**: 275줄 | **클래스**: `OCRProcessor`

#### 잘된 점 ✅
- PaddleOCR/EasyOCR 이중 엔진 지원으로 환경에 따른 유연성
- `normalize_boxes()`로 LayoutLM 입력 포맷(0-1000) 변환 자동화
- `prepare_layoutlm_input()`으로 OCR→LayoutLM 변환 원스텝 처리

#### 리뷰 이슈 🔍

**[P1] 언어 맵핑 미스**
```python
# ocr.py:39-46 (PaddleOCR)
lang_map = {"korean": "korean", ...}
# PaddleOCR 공식 문서에서는 "ko"를 사용
```
→ PaddleOCR의 한국어 코드가 `"korean"`인지 `"ko"`인지 확인 필요. (PaddleOCR 버전에 따라 다를 수 있음)

**[P2] OCR 결과 캐싱 없음**
→ 동일 이미지에 대해 `extract()`와 `prepare_layoutlm_input()`이 각각 호출되면 OCR이 2회 실행됨.

---

### 3.6 `src/step3_vlm/handler.py` & `prompts.py` — VLM 예외 처리

**파일 크기**: handler 281줄, prompts 148줄 | **클래스**: `VLMHandler`, `DocumentClassificationPrompts`

#### 잘된 점 ✅
- OpenAI/Anthropic 이중 프로바이더 지원
- 프롬프트를 별도 클래스로 분리하여 관리
- 이전 Step 결과를 프롬프트에 컨텍스트로 포함
- 응답 파싱 로직이 여러 형식을 유연하게 처리

#### 리뷰 이슈 🔍

**[P1] API 키 미설정 시 에러 메시지 부재**
```python
# handler.py:42
self.api_key = api_key or os.getenv("OPENAI_API_KEY")
# api_key가 None이어도 client 초기화 시까지 에러가 발생하지 않음
```
→ `_init_client()` 호출 시점에서야 실패. 초기화 시 키 유효성 검증 필요.

**[P2] VLM 응답 파싱의 취약성**
```python
# prompts.py:122
if line.startswith("분류:") or line.startswith("최종분류:"):
```
→ LLM 응답이 기대 형식과 다를 때 `predicted_class`가 `None`이 됨. fallback 로직이 전체 응답에서 문서명을 찾지만, "진단서를 제출하셔야 합니다" 같은 문맥에서 오탐 가능.

**[P3] 응답 신뢰도 매핑이 하드코딩**
```python
# handler.py:176
confidence_map = {"high": 0.95, "medium": 0.8, "low": 0.6}
```
→ VLM의 "높음/중간/낮음" 판단을 고정 수치로 변환. 조정 가능하도록 config로 이동 권장.

**[P4] `verify()` 메서드의 검증 로직**
```python
# handler.py:224
is_verified = "올바름" in response  # 문자열 존재 여부만으로 판단
```
→ "올바르지 않음" 같은 부정 표현에서도 "올바름"이 포함되어 True로 판단될 수 있음.

---

### 3.7 `src/preprocessor/` — 전처리 모듈

#### `deskew.py` (127줄)

**잘된 점 ✅**: Hough Transform 기반 기울기 감지, 최대 각도 제한으로 오보정 방지

**이슈 🔍**:
- **[P1]** 문서 이미지의 해상도가 매우 낮은 경우 HoughLinesP가 선분을 검출하지 못함 → angle = 0.0 반환 (silent fail)
- **[P2]** `background_color`가 BGR인지 RGB인지 명시되어 있지 않음 (OpenCV는 BGR)

#### `document_detector.py` (183줄)

**잘된 점 ✅**: Contour Detection → 4점 근사화 → Perspective Transform 파이프라인 완성

**이슈 🔍**:
- **[P1]** `detect_document_contour()`에서 4각형을 찾지 못하면 원본 반환하는데, 배경이 복잡한 이미지에서는 잘못된 영역을 검출할 수 있음
- **[P2]** `order_points()`가 볼록 사각형만 올바르게 처리 — 왜곡된 문서에서는 부정확할 수 있음

---

### 3.8 `src/utils/` — 유틸리티

#### `logger.py` (58줄)

**잘된 점 ✅**: `LoggerMixin` 패턴으로 모든 클래스에 로깅 기능 주입. 깔끔한 구현.

**이슈 🔍**:
- **[P1]** `requirements.txt`에 `loguru>=0.7.0`이 있지만, 실제 코드는 표준 `logging` 모듈 사용. loguru 의존성 불필요.
- **[P2]** 파일 핸들러가 `get_logger()`에서만 생성 가능. `LoggerMixin`은 파일 로깅을 지원하지 않음.

#### `image_utils.py` (161줄)

**잘된 점 ✅**: CV2/PIL 변환, base64 인코딩, 리사이징 등 필수 유틸리티 완비. `aspect_ratio` 옵션 제공.

---

## 4. 데이터 파이프라인 리뷰

### 4.1 합성 데이터 생성 (`scripts/generate_sample_data.py`)

**파일 크기**: 730줄 | **클래스**: `SampleDocumentGenerator`

#### 강점 ✅
- 6종 문서별로 **고유한 레이아웃 템플릿**을 코드로 생성 (진단서=텍스트중심, 보험금청구서=큰테이블+바코드 등)
- 도장, 서명, 바코드, QR코드, 테이블, 병원로고 등 요소를 실제와 유사하게 렌더링
- 가우시안 노이즈 추가로 현실감 부여
- train/test 분할 자동화

#### 데이터 규모 및 분포

| 구분 | 클래스당 | 전체 |
|------|---------|------|
| Train | 8개 | 48개 |
| Test | 2개 | 12개 |

#### 이슈 🔍

**[P1-Critical] 데이터 양 극히 부족**
→ 클래스당 8개 학습 데이터로는 일반화 불가. 특히 YOLO 학습에는 최소 수백~수천 장이 필요. 현재 **YOLO가 아무것도 검출하지 못하는 근본 원인**.

**[P2] annotations의 `boxes` 필드가 비어있음**
```json
// data/sample/train/annotations/진단서_000.json
{
  "boxes": [],  // 바운딩 박스 정보 없음!
  "words": ["진  단  서", "서울아산병원", ...],
  "label": "진단서"
}
```
→ `words`는 있지만 대응하는 `boxes`가 없어 LayoutLM 학습 시 위치 정보를 활용할 수 없음. LayoutLM의 핵심 장점인 텍스트-레이아웃 결합이 무력화.

**[P3] YOLO 학습 데이터와 샘플 데이터의 분리**
→ `data/yolo_dataset/`과 `data/sample/`이 별도로 관리되어, 데이터 일관성 확인이 어려움.

---

### 4.2 YOLO 데이터셋 (`data/yolo_dataset/data.yaml`)

```yaml
names:
  0: stamp
  1: signature
  2: table
  3: barcode
  4: qrcode
  5: hospital_logo
```

→ 6개 검출 클래스 정의. 문서 "유형"이 아닌 문서 "요소"를 검출하는 것은 올바른 설계.

---

## 5. API 서버 리뷰 (`api/app.py`)

**파일 크기**: 365줄 | **프레임워크**: FastAPI

### 5.1 엔드포인트 구성

| 엔드포인트 | 메서드 | 기능 | 상태 |
|-----------|--------|------|------|
| `/` | GET | API 정보 | ✅ |
| `/health` | GET | 헬스 체크 + 모델 상태 | ✅ |
| `/classes` | GET | 문서 클래스 목록 | ✅ |
| `/classify` | POST | 단일 문서 분류 | ✅ |
| `/classify/batch` | POST | 배치 분류 | ✅ |
| `/docs` | GET | Swagger UI (자동) | ✅ |

### 5.2 강점 ✅
- CORS 미들웨어 설정
- Pydantic 모델로 응답 스키마 명확화
- Swagger UI 자동 생성
- Mock OCR / 실제 OCR 전환 옵션

### 5.3 리뷰 이슈 🔍

**[P1-Critical] `app.py`와 `pipeline.py`의 분류 로직 중복**
```python
# app.py:242-313 — Step 1, Step 2 호출 로직이 pipeline.py와 독립적으로 구현
if yolo_classifier is not None:
    step1_result = yolo_classifier.classify(image_np)
    ...
if needs_step2:
    ...
```
→ `DocumentClassificationPipeline`을 사용하지 않고, 별도로 YOLO/LayoutLM을 직접 호출. **파이프라인 로직이 2곳에 존재**하여 불일치 위험.

**[P2] 전역 변수로 모델 관리**
```python
# app.py:70-73
yolo_detector = None
yolo_classifier = None
layoutlm_model = None
layoutlm_processor = None
```
→ 전역 변수 사용은 멀티 워커 환경에서 문제 발생 가능. FastAPI의 의존성 주입(`Depends()`) 패턴 사용 권장.

**[P3] `@app.on_event("startup")` Deprecated**
```python
# app.py:158
@app.on_event("startup")  # FastAPI에서 deprecated
```
→ FastAPI 최신 버전에서는 `lifespan` 이벤트 핸들러 사용 권장.

**[P4] 배치 API에서 파일 재사용 문제**
```python
# app.py:344-346
for file in files:
    result = await classify_document(file, use_mock_ocr, force_step2=False)
```
→ `classify_document`가 `await file.read()`를 호출하면 파일 포인터가 끝으로 이동. UploadFile의 seek(0) 없이 순차 호출하면 두 번째부터 빈 파일로 처리될 수 있음.

**[P5] 임시 파일 정리 타이밍**
```python
# app.py:327-329
finally:
    os.unlink(tmp_path)
```
→ `finally`로 정리는 좋으나, 예외 발생 시 `tmp_path` 변수가 정의되지 않을 수 있음 (try 블록 밖에서 정의되어 있어 현재는 안전하지만, 리팩토링 시 주의).

**[P6] CORS `allow_origins=["*"]`**
→ 개발 편의를 위한 설정이지만, 프로덕션 배포 시 반드시 제한해야 함.

---

## 6. 테스트 커버리지 리뷰

### 6.1 `tests/test_pipeline.py` 분석

**테스트 클래스 수**: 10개 | **테스트 메서드 수**: 22개

| 테스트 클래스 | 테스트 수 | 커버리지 범위 |
|--------------|----------|-------------|
| `TestDeskewer` | 3 | 초기화, 보정 불필요, 각도 지정 보정 |
| `TestDocumentDetector` | 3 | 초기화, 검출 실패, 점 정렬 |
| `TestYOLODetector` | 3 | 초기화, 모델 로드(모킹), 레이아웃 특징 |
| `TestYOLOClassifier` | 2 | 초기화, 점수 계산 |
| `TestOCRProcessor` | 2 | 초기화, 박스 정규화 |
| `TestLayoutLMClassifier` | 2 | 초기화, 상위 예측 |
| `TestDocumentClassificationPrompts` | 3 | 기본 프롬프트, 컨텍스트 프롬프트, 응답 파싱 |
| `TestVLMHandler` | 3 | OpenAI 초기화, Anthropic 초기화, 잘못된 프로바이더 |
| `TestClassificationResult` | 1 | to_dict() 변환 |
| `TestDocumentClassificationPipeline` | 4 | 초기화, Step 비활성화, 문서 클래스, 통계 |
| `TestIntegration` | 1 | Step1 only 파이프라인 (모킹) |

### 6.2 강점 ✅
- 모든 주요 클래스에 대한 단위 테스트 존재
- `unittest.mock` 적극 활용으로 외부 의존성 격리
- 경계값 테스트 (빈 이미지, None 결과 등)

### 6.3 커버리지 갭 ⚠️

| 미테스트 영역 | 심각도 | 설명 |
|-------------|--------|------|
| **실제 YOLO 추론** | 상 | 모킹으로만 테스트, 실제 모델 로드/추론 미검증 |
| **실제 LayoutLM 추론** | 상 | 모델 로드/전처리/추론 전체 플로우 미검증 |
| **OCR 통합** | 상 | PaddleOCR/EasyOCR 실제 실행 미검증 |
| **VLM API 호출** | 중 | 실제 API 호출 테스트 없음 (비용 문제로 합리적) |
| **에러 시나리오** | 중 | 파일 없음, 이미지 손상, 모델 파일 없음 등 |
| **API 엔드포인트** | 상 | `api/app.py`에 대한 테스트가 전혀 없음 |
| **전처리 통합** | 중 | deskew + document_detector 연계 테스트 없음 |
| **End-to-End** | 상 | 이미지 입력 → 최종 분류 전체 플로우 테스트 없음 |

### 6.4 개선 권장사항
```
1. pytest-asyncio를 도입하여 API 엔드포인트 테스트 추가
2. conftest.py에 fixture로 테스트 이미지/모델 준비
3. integration/ 디렉토리에 E2E 테스트 분리
4. CI에서 모킹 테스트와 통합 테스트를 분리 실행
```

---

## 7. 코드 품질 분석

### 7.1 코드 스타일 및 규약

| 항목 | 상태 | 비고 |
|------|------|------|
| PEP 8 준수 | ✅ 양호 | 전반적으로 잘 지켜짐 |
| Type Hints | ✅ 양호 | `Union`, `Optional`, `Dict` 등 적극 사용 |
| Docstring | ✅ 우수 | 거의 모든 메서드에 Google 스타일 docstring |
| 네이밍 | ✅ 양호 | snake_case 일관 적용 |
| 한글 주석 | ✅ 우수 | 도메인 특성에 맞는 한글 주석 |

### 7.2 설계 패턴 활용

| 패턴 | 적용 위치 | 평가 |
|------|----------|------|
| **Strategy** | OCR 엔진 선택 (paddleocr/easyocr), VLM 프로바이더 선택 | ✅ 적절 |
| **Template Method** | 각 Step의 `classify()` → `__call__()` | ✅ 적절 |
| **Mixin** | `LoggerMixin` | ✅ 적절 |
| **Chain of Responsibility** | Pipeline의 Step 1→2→3 | ⚠️ 현재는 if-else로 구현, 패턴화 가능 |
| **Factory** | 미적용 | ⚠️ config 기반 컴포넌트 생성에 Factory 패턴 도입 권장 |

### 7.3 의존성 관리 (`requirements.txt`)

#### 이슈 🔍

| 패키지 | 이슈 |
|--------|------|
| `loguru>=0.7.0` | 코드에서 사용되지 않음 (표준 logging 사용) — 제거 권장 |
| `paddleocr>=2.7.0` + `paddlepaddle>=2.5.0` | EasyOCR이 기본이므로 PaddleOCR는 선택 사항으로 변경 고려 |
| `easyocr>=1.7.0` | ✅ EasyOCR 적용 완료 — 주석 해제 필요 (requirements.txt에서 아직 주석 상태) |
| 버전 상한 없음 | `>=` 만 사용 — 호환성 문제 발생 가능. 상한 지정 또는 lock 파일 필요 |
| `uvicorn` 누락 | `run_api.py`에서 사용하지만 requirements.txt에 없음 |
| `fastapi` 누락 | `api/app.py`에서 사용하지만 requirements.txt에 없음 |
| `scikit-learn` 누락 | `train.py` evaluate에서 사용하지만 requirements.txt에 없음 |

---

## 8. 보안 및 에러 처리 리뷰

### 8.1 보안 사항

| 항목 | 상태 | 설명 |
|------|------|------|
| API 키 관리 | ⚠️ | 환경 변수 사용은 적절하나, `.env` 파일 로딩 코드 없음 (`python-dotenv` 미활용) |
| 파일 업로드 제한 | ⚠️ | 파일 확장자만 체크, 크기 제한 없음 — DoS 가능 |
| CORS 설정 | ❌ | `allow_origins=["*"]` — 프로덕션 부적합 |
| 인증/인가 | ❌ | API 인증 없음 — 누구나 접근 가능 |
| 입력 검증 | ⚠️ | 이미지 형식만 확인, 이미지 내용(크기, 메모리) 미검증 |
| 임시 파일 | ✅ | `finally` 블록에서 삭제 처리 |

### 8.2 에러 처리

| 항목 | 상태 | 설명 |
|------|------|------|
| 모델 로드 실패 | ⚠️ | 일부는 ImportError만 처리, 파일 미존재는 silent fallback |
| API 예외 | ✅ | FastAPI HTTPException 사용 |
| 이미지 로드 실패 | ✅ | FileNotFoundError, ValueError 발생 |
| VLM API 실패 | ⚠️ | 전체 예외를 catch 후 re-raise — 재시도 로직 없음 |
| OCR 실패 | ⚠️ | OCR 초기화 실패만 처리, 추출 실패는 미처리 |

---

## 9. 성능 분석

### 9.1 추론 속도 (results.json 기반)

| 이미지 | 처리 시간 | Step | 비고 |
|--------|----------|------|------|
| 첫 번째 이미지 | 1.13초 | Step 1 | 모델 로드 포함 (cold start) |
| 이후 이미지 | 0.04~0.10초 | Step 1 | 모델 캐시 후 (warm) |

→ **Cold start**: ~1.1초, **Warm**: ~0.05초 (Step 1 only). Step 2 추가 시 LayoutLM 추론 + OCR 시간이 합산되어 0.5~2초 예상.

### 9.2 메모리 사용량 예상

| 컴포넌트 | 예상 메모리 |
|----------|-----------|
| YOLOv8n | ~6MB |
| LayoutLMv3-base | ~500MB |
| PaddleOCR | ~100MB |
| EasyOCR | ~200MB |
| **합계** | **~800MB** |

### 9.3 성능 병목점

1. **OCR 처리**: 이미지당 수백ms~수초 소요 가능
2. **LayoutLM 추론**: GPU 없이 CPU에서 수백ms
3. **VLM API 호출**: 네트워크 레이턴시 + 추론 시간 (~2-5초)
4. **이미지 전처리**: Deskew의 Hough Transform이 고해상도에서 느릴 수 있음

---

## 10. 현재 결과 분석

### 10.1 `results.json` 분석 — 치명적 문제 발견 ❌

**테스트 결과 요약** (12개 테스트 이미지):

| 실제 클래스 | 예측 결과 | 신뢰도 | 정확도 |
|------------|----------|--------|--------|
| 보험금청구서 (2장) | 진단서 | 0.20 | 0% |
| 소견서 (2장) | 진단서 | 0.20 | 0% |
| 의료비영수증 (2장) | 진단서 | 0.20 | 0% |
| 입퇴원확인서 (2장) | 진단서 | 0.20 | 0% |
| 진단서 (2장) | 진단서 | 0.20 | 100% |
| 처방전 (2장) | 진단서 | 0.20 | 0% |
| **전체** | **모두 "진단서"** | **0.20** | **16.7%** |

### 10.2 근본 원인 분석

```
문제 체인:
1. YOLO 모델이 아무 객체도 검출하지 못함 (detection_count: 0)
   └─ 원인: 합성 데이터 48장으로 학습된 YOLO 모델의 검출 능력 부족
             또는 사전학습 COCO 모델 사용 (stamp, table 등 미학습 클래스)

2. 레이아웃 특징이 모두 False/0
   └─ has_stamp: false, has_table: false, has_barcode: false ...

3. YOLOClassifier 점수 계산에서 모든 클래스가 동일한 낮은 점수
   └─ 필수 요소 없음 → penalty 0.5 적용
   └─ 테이블 비율 0.0 → "진단서"(range 0.0~0.1)가 가장 유리

4. "진단서"가 weight 1.0으로 항상 최고 점수 (0.20)
   └─ "소견서"는 weight 0.95로 0.18

5. 신뢰도 0.20 < threshold 0.50 → requires_step2 = True
   └─ 그러나 Step 2 결과가 null → Step 2가 실행되지 않은 것으로 보임

6. Step 2 미실행 원인:
   └─ 파이프라인 설정에서 enable_step2=True이지만 
      LayoutLM 모델이 로드되지 않았거나 OCR 엔진 초기화 실패 가능
```

### 10.3 즉시 조치 필요 사항

1. **YOLO 모델 확인**: `data/models/yolo_document_layout/weights/best.pt`가 실제로 존재하고 합성 데이터로 학습된 모델인지 확인
2. **Step 2 실행 여부 확인**: LayoutLM 모델(`data/models/layoutlm_classifier/best`) 존재 확인 및 로그 확인
3. **에러 로그 확인**: Step 2 초기화/실행 시 발생한 예외가 로그에 기록되었는지 확인
4. **데이터 증강**: 합성 데이터를 최소 클래스당 100장 이상으로 증가

---

## 11. 종합 평가 및 개선 로드맵

### 11.1 종합 평가

| 평가 항목 | 점수 | 코멘트 |
|----------|------|--------|
| **아키텍처 설계** | ⭐⭐⭐⭐☆ | 3단계 에스컬레이션 구조 우수, 모듈화 잘 됨 |
| **코드 품질** | ⭐⭐⭐⭐☆ | Type hints, docstring, 네이밍 우수 |
| **기능 완성도** | ⭐⭐⭐☆☆ | 전체 프레임워크 완성, 실제 작동은 미흡 |
| **테스트 커버리지** | ⭐⭐⭐☆☆ | 단위 테스트 존재하나 통합/E2E 부족 |
| **모델 성능** | ⭐⭐☆☆☆ | 현재 전체 정확도 16.7% (Step 1 only) |
| **데이터 파이프라인** | ⭐⭐☆☆☆ | 합성 데이터 소량, annotation 품질 미흡 |
| **문서화** | ⭐⭐⭐⭐⭐ | README, 구현 보고서, 코드 주석 모두 우수 |
| **배포 준비도** | ⭐⭐☆☆☆ | Docker 없음, 환경 변수 관리 미흡 |

### 11.2 개선 우선순위

#### 🔴 긴급 (즉시)

| # | 작업 | 영향 |
|---|------|------|
| 1 | **Step 2 실행 안 되는 문제 디버깅** — LayoutLM 모델/OCR 로드 실패 원인 파악 및 수정 | 정확도 16.7% → 100% (합성 데이터 기준) |
| 2 | **YOLO fallback 시 경고/에러 발생**하도록 수정 — COCO 모델로 문서 분류하는 것 방지 | Silent failure 제거 |
| 3 | **DOCUMENT_CLASSES 중앙 상수화** — 6곳 중복 제거 | 유지보수성 향상 |
| 4 | **requirements.txt 누락 패키지 추가** (fastapi, uvicorn, scikit-learn) + easyocr 주석 해제 반영 | 신규 개발자 온보딩 |

#### 🟡 중요 (1-2주)

| # | 작업 | 영향 |
|---|------|------|
| 5 | **합성 데이터 증강** — 클래스당 최소 100장, bounding box annotation 추가 | 모델 성능 향상 |
| 6 | **API에서 Pipeline 클래스 사용**하도록 리팩토링 (app.py 분류 로직 중복 제거) | 코드 일관성 |
| 7 | **API 테스트 추가** — pytest-asyncio + httpx 기반 | 배포 안정성 |
| 8 | **에러 복구 로직** — Step 실패 시 다음 Step으로 자동 fallback | 안정성 |
| 9 | **`__call__` 시그니처 통일** — 특히 LayoutLMClassifier의 `__call__`에 `step1_result` 파라미터 추가 | 인터페이스 일관성 |

#### 🟢 개선 (2-4주)

| # | 작업 | 영향 |
|---|------|------|
| 10 | **Docker 컨테이너화** + docker-compose | 배포 편의성 |
| 11 | **모델 버전 관리** — 모델 파일에 메타데이터(학습일, 데이터셋, 성능) 연결 | 추적성 |
| 12 | **배치 추론 실제 구현** — GPU 배치 처리 | 처리량 향상 |
| 13 | **입력 검증 강화** — 이미지 크기/포맷 제한, API 인증 | 보안 |
| 14 | **OCR 결과 캐싱** — LRU 캐시 도입 | 성능 |
| 15 | **CI/CD 파이프라인** — GitHub Actions로 자동 테스트/빌드 | 개발 프로세스 |

---

### 11.3 리뷰어를 위한 Discussion Points

리뷰 미팅에서 논의할 핵심 질문들:

1. **데이터 전략**: 실제 의료 문서 데이터 확보 계획은? 개인정보 비식별화 방안은?
2. **Step 1-2 경계**: YOLO 신뢰도 threshold (현재 0.5)를 낮추면 더 많은 문서가 Step 2로 넘어가지만, Step 2 호출 비용이 증가. 최적 threshold는?
3. **VLM 비용**: Step 3 VLM 호출 시 건당 비용(GPT-4o ~$0.01-0.05). 월 처리량 대비 비용 수용 가능?
4. **진단서/소견서**: 레이아웃이 거의 동일한 두 문서를 Step 1에서 "미결정"으로 빠르게 넘기는 것과, 규칙을 더 세분화하는 것 중 어느 전략이 적합?
5. **실시간 vs 배치**: API 서버의 실시간 처리와 대량 배치 처리 중 어느 것이 주 유스케이스?
6. **모델 업데이트**: 신규 문서 유형 추가 시 전체 재학습? 또는 증분 학습 지원?

---

*본 리뷰 문서는 프로젝트의 전체 소스 코드, 설정, 테스트, 데이터, 실행 결과를 기반으로 작성되었습니다.*
