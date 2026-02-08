"""문서 영역 검출 모듈"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from ..utils.logger import LoggerMixin


class DocumentDetector(LoggerMixin):
    """문서 영역 검출 및 추출 클래스"""

    def __init__(self, min_area_ratio: float = 0.3):
        """
        Args:
            min_area_ratio: 최소 문서 영역 비율 (전체 이미지 대비)
        """
        self.min_area_ratio = min_area_ratio

    def detect_document_contour(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """문서 윤곽선 검출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            문서 윤곽선 좌표 (4개 꼭지점) 또는 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self.logger.debug("윤곽선을 찾지 못했습니다.")
            return None

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    valid_contours.append((approx, area))

        if not valid_contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:5]:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    epsilon = 0.05 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        return approx.reshape(4, 2)

            self.logger.debug("4각형 문서 영역을 찾지 못했습니다.")
            return None

        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        return best_contour.reshape(4, 2)

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """4개 꼭지점을 [좌상, 우상, 우하, 좌하] 순서로 정렬

        Args:
            pts: 4개 꼭지점 좌표

        Returns:
            정렬된 좌표
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def perspective_transform(
        self,
        image: np.ndarray,
        pts: np.ndarray
    ) -> np.ndarray:
        """원근 변환으로 문서 추출

        Args:
            image: 입력 이미지
            pts: 4개 꼭지점 좌표

        Returns:
            추출된 문서 이미지
        """
        rect = self.order_points(pts.astype(np.float32))
        tl, tr, br, bl = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        return warped

    def detect_and_extract(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """문서 영역 검출 및 추출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            (추출된 문서 이미지, 검출된 좌표) 튜플
            문서를 찾지 못한 경우 원본 이미지와 None 반환
        """
        contour = self.detect_document_contour(image)

        if contour is None:
            self.logger.info("문서 영역을 찾지 못해 원본 이미지를 반환합니다.")
            return image, None

        extracted = self.perspective_transform(image, contour)
        self.logger.info(f"문서 영역 추출 완료: {extracted.shape[1]}x{extracted.shape[0]}")

        return extracted, contour

    def get_bounding_box(
        self,
        contour: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """윤곽선의 바운딩 박스 반환

        Args:
            contour: 윤곽선 좌표

        Returns:
            (x, y, width, height) 튜플
        """
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h

    def __call__(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """detect_and_extract 메서드 호출"""
        return self.detect_and_extract(image)
