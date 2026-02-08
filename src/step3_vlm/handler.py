"""VLM 기반 예외 처리 모듈"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

from .prompts import DocumentClassificationPrompts
from ..utils.logger import LoggerMixin
from ..utils.image_utils import encode_image_base64


class VLMHandler(LoggerMixin):
    """VLM 기반 문서 분류 예외 처리기"""

    SUPPORTED_PROVIDERS = ["openai", "anthropic"]

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 500
    ):
        """
        Args:
            provider: VLM 제공자 ("openai" 또는 "anthropic")
            model: 사용할 모델 이름
            api_key: API 키 (None이면 환경 변수에서 로드)
            max_tokens: 최대 응답 토큰 수
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"지원하지 않는 provider입니다: {provider}")

        self.provider = provider
        self.max_tokens = max_tokens

        if provider == "openai":
            self.model = model or "gpt-4o"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.model = model or "claude-3-5-sonnet-20241022"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        self.client = None

    def _init_client(self) -> None:
        """API 클라이언트 초기화"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai 패키지를 설치해주세요: pip install openai")

            self.client = OpenAI(api_key=self.api_key)

        else:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic 패키지를 설치해주세요: pip install anthropic")

            self.client = anthropic.Anthropic(api_key=self.api_key)

        self.logger.info(f"VLM 클라이언트 초기화: {self.provider}/{self.model}")

    def _call_openai(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str
    ) -> str:
        """OpenAI API 호출"""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    def _call_anthropic(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str
    ) -> str:
        """Anthropic API 호출"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        return response.content[0].text

    def classify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        step1_result: Optional[Dict] = None,
        step2_result: Optional[Dict] = None
    ) -> Dict:
        """VLM을 사용한 문서 분류

        Args:
            image: 입력 이미지
            step1_result: Step 1 분류 결과 (선택)
            step2_result: Step 2 분류 결과 (선택)

        Returns:
            분류 결과 딕셔너리:
            {
                "predicted_class": str,
                "confidence_level": str,
                "reasoning": str,
                "raw_response": str,
                "provider": str,
                "model": str
            }
        """
        if self.client is None:
            self._init_client()

        image_base64 = encode_image_base64(image)

        prompt = DocumentClassificationPrompts.get_classification_prompt(
            step1_result, step2_result
        )
        system_prompt = DocumentClassificationPrompts.SYSTEM_PROMPT

        self.logger.info(f"VLM 분류 요청: {self.provider}/{self.model}")

        try:
            if self.provider == "openai":
                response = self._call_openai(image_base64, prompt, system_prompt)
            else:
                response = self._call_anthropic(image_base64, prompt, system_prompt)

            result = DocumentClassificationPrompts.parse_response(response)

            result["provider"] = self.provider
            result["model"] = self.model

            confidence_map = {"high": 0.95, "medium": 0.8, "low": 0.6}
            result["confidence"] = confidence_map.get(
                result.get("confidence_level"), 0.7
            )

            self.logger.info(
                f"VLM 분류 결과: {result['predicted_class']} "
                f"(신뢰도: {result['confidence_level']})"
            )

            return result

        except Exception as e:
            self.logger.error(f"VLM API 호출 실패: {e}")
            raise

    def verify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        predicted_class: str
    ) -> Dict:
        """VLM을 사용한 분류 검증

        Args:
            image: 입력 이미지
            predicted_class: 검증할 예측 클래스

        Returns:
            검증 결과 딕셔너리
        """
        if self.client is None:
            self._init_client()

        image_base64 = encode_image_base64(image)

        prompt = DocumentClassificationPrompts.get_verification_prompt(predicted_class)
        system_prompt = DocumentClassificationPrompts.SYSTEM_PROMPT

        self.logger.info(f"VLM 검증 요청: {predicted_class}")

        try:
            if self.provider == "openai":
                response = self._call_openai(image_base64, prompt, system_prompt)
            else:
                response = self._call_anthropic(image_base64, prompt, system_prompt)

            result = DocumentClassificationPrompts.parse_response(response)

            is_verified = "올바름" in response
            result["is_verified"] = is_verified
            result["original_prediction"] = predicted_class

            if is_verified:
                result["predicted_class"] = predicted_class

            self.logger.info(
                f"VLM 검증 결과: {'확인됨' if is_verified else '수정됨'} "
                f"-> {result['predicted_class']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"VLM API 호출 실패: {e}")
            raise

    def batch_classify(
        self,
        images: list,
        step_results: Optional[list] = None
    ) -> list:
        """여러 이미지 일괄 분류

        Args:
            images: 이미지 리스트
            step_results: 이전 단계 결과 리스트 (선택)

        Returns:
            분류 결과 리스트
        """
        results = []

        for i, image in enumerate(images):
            self.logger.debug(f"VLM 처리 중: {i + 1}/{len(images)}")

            step1_result = None
            step2_result = None

            if step_results and i < len(step_results):
                step1_result = step_results[i].get("step1")
                step2_result = step_results[i].get("step2")

            result = self.classify(image, step1_result, step2_result)
            results.append(result)

        return results

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        step1_result: Optional[Dict] = None,
        step2_result: Optional[Dict] = None
    ) -> Dict:
        """classify 메서드 호출"""
        return self.classify(image, step1_result, step2_result)
