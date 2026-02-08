"""VLM 프롬프트 템플릿"""

from typing import Dict, List, Optional


class DocumentClassificationPrompts:
    """문서 분류용 VLM 프롬프트"""

    DOCUMENT_CLASSES = [
        "진단서",
        "소견서",
        "보험금청구서",
        "입퇴원확인서",
        "의료비영수증",
        "처방전"
    ]

    SYSTEM_PROMPT = """당신은 의료 및 보험 관련 문서를 분류하는 전문가입니다.
주어진 문서 이미지를 분석하여 다음 카테고리 중 하나로 정확하게 분류해주세요.

분류 카테고리:
1. 진단서: 의사가 환자의 질병이나 상해에 대해 진단 내용을 기재한 문서
2. 소견서: 의사가 환자의 상태에 대한 의학적 소견을 기재한 문서
3. 보험금청구서: 보험금 지급을 요청하는 양식 문서
4. 입퇴원확인서: 환자의 입원 및 퇴원 사실을 확인하는 문서
5. 의료비영수증: 의료비 지불 내역을 나타내는 영수증
6. 처방전: 의사가 환자에게 처방하는 약품 정보가 기재된 문서

반드시 위 6개 카테고리 중 하나로만 분류해주세요."""

    CLASSIFICATION_PROMPT = """이 문서 이미지를 분석하여 문서 유형을 분류해주세요.

다음 형식으로 응답해주세요:
분류: [문서 유형]
신뢰도: [높음/중간/낮음]
근거: [분류 근거를 간단히 설명]

분류 가능한 문서 유형: 진단서, 소견서, 보험금청구서, 입퇴원확인서, 의료비영수증, 처방전"""

    CLASSIFICATION_WITH_CONTEXT_PROMPT = """이 문서 이미지를 분석하여 문서 유형을 분류해주세요.

이전 분석 정보:
- 1차 분류 결과: {step1_class} (신뢰도: {step1_confidence:.2f})
- 2차 분류 결과: {step2_class} (신뢰도: {step2_confidence:.2f})

위 정보를 참고하되, 이미지를 직접 분석하여 최종 분류를 결정해주세요.

다음 형식으로 응답해주세요:
분류: [문서 유형]
신뢰도: [높음/중간/낮음]
근거: [분류 근거를 간단히 설명]

분류 가능한 문서 유형: 진단서, 소견서, 보험금청구서, 입퇴원확인서, 의료비영수증, 처방전"""

    VERIFICATION_PROMPT = """이전 시스템에서 이 문서를 '{predicted_class}'로 분류했습니다.

이 분류가 올바른지 검증해주세요.

다음 형식으로 응답해주세요:
검증결과: [올바름/틀림]
최종분류: [문서 유형]
근거: [판단 근거]"""

    @classmethod
    def get_classification_prompt(
        cls,
        step1_result: Optional[Dict] = None,
        step2_result: Optional[Dict] = None
    ) -> str:
        """분류 프롬프트 생성

        Args:
            step1_result: Step 1 분류 결과
            step2_result: Step 2 분류 결과

        Returns:
            프롬프트 문자열
        """
        if step1_result and step2_result:
            return cls.CLASSIFICATION_WITH_CONTEXT_PROMPT.format(
                step1_class=step1_result.get("predicted_class", "N/A"),
                step1_confidence=step1_result.get("confidence", 0.0),
                step2_class=step2_result.get("predicted_class", "N/A"),
                step2_confidence=step2_result.get("confidence", 0.0)
            )
        return cls.CLASSIFICATION_PROMPT

    @classmethod
    def get_verification_prompt(cls, predicted_class: str) -> str:
        """검증 프롬프트 생성

        Args:
            predicted_class: 검증할 예측 클래스

        Returns:
            프롬프트 문자열
        """
        return cls.VERIFICATION_PROMPT.format(predicted_class=predicted_class)

    @classmethod
    def parse_response(cls, response: str) -> Dict:
        """VLM 응답 파싱

        Args:
            response: VLM 응답 문자열

        Returns:
            파싱된 결과 딕셔너리
        """
        result = {
            "predicted_class": None,
            "confidence_level": None,
            "reasoning": None,
            "raw_response": response
        }

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("분류:") or line.startswith("최종분류:"):
                class_text = line.split(":", 1)[1].strip()
                for doc_class in cls.DOCUMENT_CLASSES:
                    if doc_class in class_text:
                        result["predicted_class"] = doc_class
                        break

            elif line.startswith("신뢰도:"):
                confidence_text = line.split(":", 1)[1].strip()
                if "높음" in confidence_text:
                    result["confidence_level"] = "high"
                elif "중간" in confidence_text:
                    result["confidence_level"] = "medium"
                elif "낮음" in confidence_text:
                    result["confidence_level"] = "low"

            elif line.startswith("근거:") or line.startswith("판단 근거:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        if result["predicted_class"] is None:
            for doc_class in cls.DOCUMENT_CLASSES:
                if doc_class in response:
                    result["predicted_class"] = doc_class
                    break

        return result
