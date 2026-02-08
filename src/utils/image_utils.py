"""이미지 처리 유틸리티"""

import base64
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(
    image_path: Union[str, Path],
    mode: str = "cv2"
) -> Union[np.ndarray, Image.Image]:
    """이미지 로드

    Args:
        image_path: 이미지 파일 경로
        mode: 반환 모드 ("cv2" 또는 "pil")

    Returns:
        로드된 이미지 (numpy array 또는 PIL Image)
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    if mode == "cv2":
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        return image
    elif mode == "pil":
        return Image.open(image_path).convert("RGB")
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")


def save_image(
    image: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """이미지 저장

    Args:
        image: 저장할 이미지
        output_path: 저장 경로
        quality: JPEG 품질 (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif isinstance(image, Image.Image):
        image.save(output_path, quality=quality)
    else:
        raise TypeError(f"지원하지 않는 이미지 타입입니다: {type(image)}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """이미지 크기 조정

    Args:
        image: 입력 이미지
        target_size: 목표 크기 (width, height)
        max_size: 최대 크기 (긴 변 기준)
        keep_aspect_ratio: 종횡비 유지 여부

    Returns:
        크기가 조정된 이미지
    """
    h, w = image.shape[:2]

    if max_size is not None:
        scale = max_size / max(h, w)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    if target_size is not None:
        target_w, target_h = target_size

        if keep_aspect_ratio:
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = target_w, target_h

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV 이미지를 PIL 이미지로 변환"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """PIL 이미지를 OpenCV 이미지로 변환"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode_image_base64(image: Union[np.ndarray, Image.Image, str, Path]) -> str:
    """이미지를 base64 문자열로 인코딩

    Args:
        image: 이미지 (numpy array, PIL Image, 또는 파일 경로)

    Returns:
        base64 인코딩된 문자열
    """
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if isinstance(image, np.ndarray):
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    if isinstance(image, Image.Image):
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    raise TypeError(f"지원하지 않는 이미지 타입입니다: {type(image)}")


def get_image_info(image: np.ndarray) -> dict:
    """이미지 정보 반환

    Args:
        image: 입력 이미지

    Returns:
        이미지 정보 딕셔너리
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes
    }
