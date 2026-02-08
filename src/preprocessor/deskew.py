"""기울기 보정 모듈"""

import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import LoggerMixin


class Deskewer(LoggerMixin):
    """문서 이미지 기울기 보정 클래스"""

    def __init__(self, max_angle: float = 15.0):
        """
        Args:
            max_angle: 보정할 최대 각도 (도 단위)
        """
        self.max_angle = max_angle

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """문서 기울기 각도 검출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            검출된 기울기 각도 (도 단위)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            self.logger.debug("직선을 검출하지 못했습니다.")
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(angle) < self.max_angle:
                angles.append(angle)
            elif abs(angle - 90) < self.max_angle:
                angles.append(angle - 90)
            elif abs(angle + 90) < self.max_angle:
                angles.append(angle + 90)

        if not angles:
            return 0.0

        median_angle = np.median(angles)
        self.logger.debug(f"검출된 기울기 각도: {median_angle:.2f}°")

        return float(median_angle)

    def deskew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[np.ndarray, float]:
        """이미지 기울기 보정

        Args:
            image: 입력 이미지 (BGR)
            angle: 보정할 각도 (None이면 자동 검출)
            background_color: 회전 후 빈 영역 채울 색상

        Returns:
            (보정된 이미지, 보정 각도) 튜플
        """
        if angle is None:
            angle = self.detect_skew_angle(image)

        if abs(angle) < 0.1:
            self.logger.info("기울기 보정이 필요하지 않습니다.")
            return image, 0.0

        if abs(angle) > self.max_angle:
            self.logger.warning(
                f"검출된 각도({angle:.2f}°)가 최대 허용 각도({self.max_angle}°)를 초과합니다."
            )
            return image, 0.0

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )

        self.logger.info(f"이미지 기울기 보정 완료: {angle:.2f}°")

        return rotated, angle

    def __call__(
        self,
        image: np.ndarray,
        angle: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """deskew 메서드 호출"""
        return self.deskew(image, angle)
