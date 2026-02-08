#!/usr/bin/env python3
"""테스트용 샘플 문서 이미지 생성 스크립트 - 문서 유형별 다른 레이아웃"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SampleDocumentGenerator:
    """샘플 문서 이미지 생성기 - 문서 유형별 고유 레이아웃"""

    HOSPITAL_NAMES = [
        "서울대학교병원", "연세대학교 세브란스병원", "삼성서울병원",
        "서울아산병원", "고려대학교병원", "한양대학교병원"
    ]

    INSURANCE_COMPANIES = [
        "삼성화재", "현대해상", "DB손해보험", "KB손해보험", "메리츠화재"
    ]

    PATIENT_NAMES = ["김철수", "이영희", "박민수", "정수진", "최준혁", "강미래"]

    MEDICINE_NAMES = [
        "타이레놀정 500mg", "아스피린정 100mg", "오메프라졸캡슐 20mg",
        "메트포르민정 500mg", "아토르바스타틴정 10mg", "암로디핀정 5mg"
    ]

    def __init__(self, output_dir: str = "data/sample"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_cache = {}

    def _get_font(self, size: int = 20):
        """폰트 로드 (캐싱)"""
        if size in self.font_cache:
            return self.font_cache[size]

        font_paths = [
            "/mnt/c/Windows/Fonts/malgun.ttf",
            "/mnt/c/Windows/Fonts/malgunbd.ttf",
            "/mnt/c/Windows/Fonts/gulim.ttc",
            "C:/Windows/Fonts/malgun.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]

        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size)
                self.font_cache[size] = font
                return font
            except (OSError, IOError):
                continue

        return ImageFont.load_default()

    def _draw_stamp(self, draw: ImageDraw, x: int, y: int, size: int = 50, text: str = "인"):
        """도장 그리기"""
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            outline="red", width=3
        )
        font = self._get_font(size // 2 + 5)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x - text_w // 2, y - text_h // 2), text, fill="red", font=font)

    def _draw_signature(self, draw: ImageDraw, x: int, y: int, width: int = 100):
        """서명 그리기"""
        points = []
        for i in range(25):
            px = x + i * (width // 25) + random.randint(-2, 2)
            py = y + random.randint(-8, 8) + int(8 * np.sin(i / 2.5))
            points.append((px, py))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill="darkblue", width=2)

    def _draw_table(self, draw: ImageDraw, x: int, y: int,
                    rows: int, cols: int, cell_w: int = 100, cell_h: int = 30,
                    headers: List[str] = None, data: List[List[str]] = None):
        """테이블 그리기"""
        font = self._get_font(12)

        # 테이블 외곽선
        draw.rectangle(
            [x, y, x + cols * cell_w, y + rows * cell_h],
            outline="black", width=2
        )

        # 가로선
        for i in range(rows + 1):
            line_width = 2 if i == 0 or i == 1 else 1
            draw.line([(x, y + i * cell_h), (x + cols * cell_w, y + i * cell_h)],
                     fill="black", width=line_width)

        # 세로선
        for j in range(cols + 1):
            draw.line([(x + j * cell_w, y), (x + j * cell_w, y + rows * cell_h)],
                     fill="black", width=1)

        # 헤더
        if headers:
            for j, header in enumerate(headers[:cols]):
                draw.text((x + j * cell_w + 5, y + 5), header, fill="black", font=font)

        # 데이터
        if data:
            for i, row_data in enumerate(data[:rows-1]):
                for j, cell in enumerate(row_data[:cols]):
                    draw.text((x + j * cell_w + 5, y + (i + 1) * cell_h + 5),
                             str(cell), fill="black", font=font)

    def _draw_barcode(self, draw: ImageDraw, x: int, y: int, width: int = 150, height: int = 40):
        """바코드 그리기"""
        bar_x = x
        while bar_x < x + width:
            bar_width = random.choice([2, 3, 4])
            if random.random() > 0.5:
                draw.rectangle([bar_x, y, bar_x + bar_width, y + height], fill="black")
            bar_x += bar_width + random.randint(1, 3)

        # 바코드 번호
        font = self._get_font(10)
        barcode_num = ''.join([str(random.randint(0, 9)) for _ in range(13)])
        draw.text((x + 20, y + height + 2), barcode_num, fill="black", font=font)

    def _draw_qrcode(self, draw: ImageDraw, x: int, y: int, size: int = 60):
        """QR코드 모양 그리기"""
        cell_size = size // 10

        # 외곽 사각형
        draw.rectangle([x, y, x + size, y + size], outline="black", width=2)

        # 랜덤 패턴
        for i in range(10):
            for j in range(10):
                if random.random() > 0.5:
                    draw.rectangle(
                        [x + i * cell_size, y + j * cell_size,
                         x + (i + 1) * cell_size, y + (j + 1) * cell_size],
                        fill="black"
                    )

        # 위치 탐지 패턴 (모서리 3개)
        for px, py in [(x, y), (x + size - 21, y), (x, y + size - 21)]:
            draw.rectangle([px, py, px + 21, py + 21], fill="black")
            draw.rectangle([px + 3, py + 3, px + 18, py + 18], fill="white")
            draw.rectangle([px + 6, py + 6, px + 15, py + 15], fill="black")

    def _draw_hospital_logo(self, draw: ImageDraw, x: int, y: int, name: str, style: str = "box"):
        """병원 로고 영역 그리기"""
        font = self._get_font(14)

        if style == "box":
            draw.rectangle([x, y, x + 180, y + 50], outline="navy", width=2)
            draw.text((x + 10, y + 15), name, fill="navy", font=font)
        elif style == "underline":
            draw.text((x, y), name, fill="navy", font=self._get_font(18))
            bbox = draw.textbbox((x, y), name, font=self._get_font(18))
            draw.line([(x, bbox[3] + 2), (bbox[2], bbox[3] + 2)], fill="navy", width=2)
        elif style == "circle":
            draw.ellipse([x, y, x + 40, y + 40], outline="navy", width=2)
            draw.text((x + 50, y + 10), name, fill="navy", font=font)

    # ==================== 문서 유형별 생성 함수 ====================

    def generate_diagnosis(self, index: int) -> Tuple[Image.Image, dict]:
        """진단서 생성 - 텍스트 중심, 테이블 없음, 도장+서명"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "진단서"}

        # 제목 (큰 글씨, 중앙)
        title_font = self._get_font(32)
        title = "진  단  서"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((title_x, 50), title, fill="black", font=title_font)
        annotations["words"].append(title)

        # 병원 로고 (좌상단, 박스 스타일)
        hospital = random.choice(self.HOSPITAL_NAMES)
        self._draw_hospital_logo(draw, 50, 110, hospital, "box")
        annotations["words"].append(hospital)

        # 환자 정보 (텍스트 필드들)
        field_font = self._get_font(16)
        patient = random.choice(self.PATIENT_NAMES)
        fields = [
            ("환자 성명", patient),
            ("주민등록번호", f"{random.randint(700000, 999999)}-*******"),
            ("주      소", "서울특별시 강남구 테헤란로 123"),
            ("병      명", random.choice(["급성 상기도 감염", "요추 염좌", "위염", "고혈압"])),
            ("진 단  일", f"2024.{random.randint(1,12):02d}.{random.randint(1,28):02d}"),
            ("치료기간", f"{random.randint(7, 30)}일"),
            ("비      고", ""),
        ]

        y_pos = 200
        for label, value in fields:
            draw.text((80, y_pos), f"{label}:", fill="black", font=field_font)
            draw.text((220, y_pos), value, fill="black", font=field_font)
            annotations["words"].extend([label, value])
            y_pos += 45

        # 소견 텍스트 영역
        draw.text((80, y_pos + 20), "위 환자는 상기 병명으로 진단하였기에 이에 확인합니다.",
                 fill="black", font=field_font)
        annotations["words"].append("위 환자는 상기 병명으로 진단하였기에 이에 확인합니다.")

        # 날짜
        draw.text((width // 2 - 80, height - 250),
                 f"2024년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                 fill="black", font=field_font)

        # 병원명 + 의사
        draw.text((width // 2 - 80, height - 200), hospital, fill="black", font=field_font)
        draw.text((width // 2 - 80, height - 160), "담당의사: 홍길동 (인)", fill="black", font=field_font)

        # 서명 (좌하단)
        self._draw_signature(draw, width - 320, height - 150, 100)

        # 도장 (우하단, 큰 도장)
        self._draw_stamp(draw, width - 120, height - 180, 55, "인")

        return self._add_noise(image), annotations

    def generate_opinion(self, index: int) -> Tuple[Image.Image, dict]:
        """소견서 생성 - 긴 텍스트 영역, 테이블 없음"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "소견서"}

        # 제목
        title_font = self._get_font(32)
        title = "소  견  서"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, 50), title, fill="black", font=title_font)
        annotations["words"].append(title)

        # 병원 로고 (밑줄 스타일)
        hospital = random.choice(self.HOSPITAL_NAMES)
        self._draw_hospital_logo(draw, 50, 110, hospital, "underline")
        annotations["words"].append(hospital)

        # 환자 정보 (간단히)
        field_font = self._get_font(16)
        patient = random.choice(self.PATIENT_NAMES)
        draw.text((50, 170), f"환자성명: {patient}", fill="black", font=field_font)
        draw.text((400, 170), f"생년월일: 19{random.randint(60,99)}.{random.randint(1,12):02d}.{random.randint(1,28):02d}",
                 fill="black", font=field_font)
        draw.text((50, 210), f"진 료 과: {random.choice(['내과', '외과', '정형외과', '신경외과'])}",
                 fill="black", font=field_font)
        annotations["words"].extend([patient, "환자성명", "생년월일", "진료과"])

        # 구분선
        draw.line([(50, 250), (width - 50, 250)], fill="gray", width=1)

        # 소견 내용 (긴 텍스트 박스)
        draw.text((50, 270), "[ 소 견 ]", fill="black", font=self._get_font(18))

        opinion_text = [
            "상기 환자는 본원에서 진료를 받은 환자로서,",
            f"주호소: {random.choice(['요통', '두통', '복통', '관절통'])}을 호소하며 내원하였습니다.",
            "",
            f"검사 결과 {random.choice(['경미한 이상 소견', '특이 소견 없음', '추가 검사 필요'])}이",
            "확인되었으며, 현재 약물 치료 및 물리치료를 병행 중입니다.",
            "",
            "향후 치료 계획:",
            f"- {random.choice(['약물치료 지속', '물리치료 권고', '정기 검진 필요'])}",
            f"- 예상 치료 기간: 약 {random.randint(2, 8)}주",
        ]

        y_pos = 310
        small_font = self._get_font(14)
        for line in opinion_text:
            draw.text((70, y_pos), line, fill="black", font=small_font)
            if line:
                annotations["words"].append(line)
            y_pos += 30

        # 구분선
        draw.line([(50, y_pos + 20), (width - 50, y_pos + 20)], fill="gray", width=1)

        # 날짜 및 의사 정보
        draw.text((width // 2 - 100, height - 200),
                 f"2024년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                 fill="black", font=field_font)
        draw.text((width // 2 - 100, height - 160), f"담당의: 김의사", fill="black", font=field_font)

        # 서명
        self._draw_signature(draw, width - 280, height - 140, 80)

        # 도장
        self._draw_stamp(draw, width - 100, height - 160, 45, "인")

        return self._add_noise(image), annotations

    def generate_insurance_claim(self, index: int) -> Tuple[Image.Image, dict]:
        """보험금청구서 생성 - 보험사 양식, 큰 테이블, 바코드"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "보험금청구서"}

        # 보험사 로고 (좌상단)
        insurance_co = random.choice(self.INSURANCE_COMPANIES)
        draw.rectangle([30, 30, 200, 80], outline="darkgreen", width=2)
        draw.text((45, 45), insurance_co, fill="darkgreen", font=self._get_font(18))
        annotations["words"].append(insurance_co)

        # 제목
        title_font = self._get_font(28)
        title = "보험금 청구서"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, 100), title, fill="black", font=title_font)
        annotations["words"].append(title)

        # 바코드 (우상단)
        self._draw_barcode(draw, width - 200, 40, 160, 35)
        annotations["words"].append("바코드")

        # 청구인 정보 섹션
        field_font = self._get_font(14)
        draw.rectangle([40, 160, width - 40, 280], outline="black", width=1)
        draw.text((50, 165), "[ 청구인 정보 ]", fill="black", font=self._get_font(12))

        patient = random.choice(self.PATIENT_NAMES)
        info_fields = [
            ("청구인", patient), ("연락처", f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"),
            ("피보험자", patient), ("증권번호", f"INS-2024-{random.randint(100000, 999999)}"),
        ]
        y_pos = 195
        for i, (label, value) in enumerate(info_fields):
            x_pos = 60 if i % 2 == 0 else 420
            if i % 2 == 0 and i > 0:
                y_pos += 35
            draw.text((x_pos, y_pos), f"{label}: {value}", fill="black", font=field_font)
            annotations["words"].extend([label, value])

        # 사고/질병 정보
        draw.rectangle([40, 300, width - 40, 400], outline="black", width=1)
        draw.text((50, 305), "[ 사고/질병 정보 ]", fill="black", font=self._get_font(12))
        draw.text((60, 335), f"사고일자: 2024.{random.randint(1,12):02d}.{random.randint(1,28):02d}",
                 fill="black", font=field_font)
        draw.text((300, 335), f"사고유형: {random.choice(['질병', '상해', '교통사고'])}",
                 fill="black", font=field_font)
        draw.text((60, 365), f"청구사유: {random.choice(['입원비', '수술비', '통원비', '후유장해'])}",
                 fill="black", font=field_font)
        annotations["words"].extend(["사고일자", "사고유형", "청구사유"])

        # 청구 내역 테이블 (큰 테이블)
        draw.text((50, 420), "[ 청구 내역 ]", fill="black", font=self._get_font(12))
        headers = ["항목", "청구금액", "지급금액", "비고"]
        data = [
            ["입원비", f"{random.randint(100, 500) * 10000:,}원", "", ""],
            ["수술비", f"{random.randint(50, 300) * 10000:,}원", "", ""],
            ["약제비", f"{random.randint(10, 50) * 10000:,}원", "", ""],
            ["검사비", f"{random.randint(20, 100) * 10000:,}원", "", ""],
            ["합  계", f"{random.randint(200, 900) * 10000:,}원", "", ""],
        ]
        self._draw_table(draw, 50, 450, 6, 4, cell_w=175, cell_h=35, headers=headers, data=data)
        annotations["words"].extend(headers)

        # 서명란
        draw.text((50, 700), "위와 같이 보험금을 청구합니다.", fill="black", font=field_font)
        draw.text((50, 750), f"청구일: 2024년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                 fill="black", font=field_font)
        draw.text((50, 790), f"청구인: {patient} (서명 또는 인)", fill="black", font=field_font)

        # 서명
        self._draw_signature(draw, 250, 785, 100)

        # 도장
        self._draw_stamp(draw, 400, 785, 40, "인")

        # 하단 안내문
        draw.rectangle([40, height - 150, width - 40, height - 50], fill="#f0f0f0")
        small_font = self._get_font(10)
        draw.text((50, height - 140), "※ 구비서류: 진단서, 진료비 영수증, 통장사본", fill="gray", font=small_font)
        draw.text((50, height - 120), "※ 문의: 고객센터 1588-XXXX", fill="gray", font=small_font)

        return self._add_noise(image), annotations

    def generate_admission_discharge(self, index: int) -> Tuple[Image.Image, dict]:
        """입퇴원확인서 생성 - 입퇴원 정보 테이블, 간단한 구조"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "입퇴원확인서"}

        # 병원 로고 (원형 스타일 + 이름)
        hospital = random.choice(self.HOSPITAL_NAMES)
        self._draw_hospital_logo(draw, 50, 40, hospital, "circle")
        annotations["words"].append(hospital)

        # 제목 (굵은 밑줄)
        title_font = self._get_font(30)
        title = "입퇴원 확인서"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((title_x, 100), title, fill="black", font=title_font)
        draw.line([(title_x, 140), (title_x + bbox[2] - bbox[0], 140)], fill="black", width=3)
        annotations["words"].append(title)

        # 환자 정보 테이블
        field_font = self._get_font(14)
        patient = random.choice(self.PATIENT_NAMES)

        # 환자정보 테이블
        draw.text((50, 170), "■ 환자 정보", fill="black", font=self._get_font(14))
        headers1 = ["성    명", "등록번호", "생년월일"]
        data1 = [[patient, f"{random.randint(10000000, 99999999)}", f"19{random.randint(60,99)}.{random.randint(1,12):02d}.{random.randint(1,28):02d}"]]
        self._draw_table(draw, 50, 200, 2, 3, cell_w=230, cell_h=40, headers=headers1, data=data1)
        annotations["words"].extend([patient, "성명", "등록번호", "생년월일"])

        # 입퇴원 정보 테이블
        draw.text((50, 310), "■ 입퇴원 정보", fill="black", font=self._get_font(14))
        admission_date = f"2024.{random.randint(1,12):02d}.{random.randint(1,15):02d}"
        discharge_date = f"2024.{random.randint(1,12):02d}.{random.randint(16,28):02d}"
        days = random.randint(3, 14)

        headers2 = ["구분", "일자", "병동", "진료과"]
        data2 = [
            ["입원", admission_date, f"{random.randint(1,10)}병동", random.choice(["내과", "외과", "정형외과"])],
            ["퇴원", discharge_date, "", ""],
            ["재원일수", f"{days}일", "", ""],
        ]
        self._draw_table(draw, 50, 340, 4, 4, cell_w=175, cell_h=40, headers=headers2, data=data2)
        annotations["words"].extend(["입원", "퇴원", admission_date, discharge_date])

        # 진단명
        draw.text((50, 520), "■ 진단명", fill="black", font=self._get_font(14))
        draw.rectangle([50, 550, width - 50, 620], outline="black", width=1)
        diagnosis = random.choice(["급성 충수염", "요추 추간판 탈출증", "폐렴", "골절"])
        draw.text((70, 575), diagnosis, fill="black", font=field_font)
        annotations["words"].append(diagnosis)

        # 확인 문구
        draw.text((50, 660), "위 환자가 상기 기간 동안 본원에 입원하여 치료받았음을 확인합니다.",
                 fill="black", font=field_font)

        # 발급 정보
        draw.text((width // 2 - 100, 750),
                 f"2024년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                 fill="black", font=field_font)
        draw.text((width // 2 - 100, 800), hospital, fill="black", font=self._get_font(16))
        draw.text((width // 2 - 100, 840), "원 장  홍 길 동", fill="black", font=field_font)

        # 도장 (큰 직인)
        self._draw_stamp(draw, width - 150, 820, 60, "직인")

        # 하단 안내
        draw.line([(50, height - 100), (width - 50, height - 100)], fill="gray", width=1)
        draw.text((50, height - 80), f"발급번호: ADM-2024-{random.randint(10000, 99999)}",
                 fill="gray", font=self._get_font(10))

        return self._add_noise(image), annotations

    def generate_medical_receipt(self, index: int) -> Tuple[Image.Image, dict]:
        """의료비영수증 생성 - 상세 금액 테이블, 바코드"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "의료비영수증"}

        # 병원 로고
        hospital = random.choice(self.HOSPITAL_NAMES)
        draw.rectangle([50, 30, 250, 80], fill="#e8f4f8", outline="#2196F3", width=2)
        draw.text((60, 45), hospital, fill="#1976D2", font=self._get_font(14))
        annotations["words"].append(hospital)

        # 제목 (박스 강조)
        draw.rectangle([250, 90, 550, 140], fill="#1976D2")
        draw.text((290, 100), "진료비 영수증", fill="white", font=self._get_font(26))
        annotations["words"].append("진료비 영수증")

        # 바코드 (우상단)
        self._draw_barcode(draw, width - 180, 30, 140, 30)

        # 영수증 번호
        receipt_no = f"R{random.randint(20240001, 20249999)}"
        draw.text((width - 180, 80), f"No. {receipt_no}", fill="black", font=self._get_font(12))
        annotations["words"].append(receipt_no)

        # 환자 정보
        field_font = self._get_font(13)
        patient = random.choice(self.PATIENT_NAMES)
        draw.text((50, 160), f"환자명: {patient}", fill="black", font=field_font)
        draw.text((300, 160), f"등록번호: {random.randint(10000000, 99999999)}", fill="black", font=field_font)
        draw.text((50, 190), f"진료일: 2024.{random.randint(1,12):02d}.{random.randint(1,28):02d}", fill="black", font=field_font)
        draw.text((300, 190), f"진료과: {random.choice(['내과', '외과', '소아과'])}", fill="black", font=field_font)
        annotations["words"].extend([patient, "환자명", "등록번호", "진료일"])

        # 구분선
        draw.line([(50, 220), (width - 50, 220)], fill="#1976D2", width=2)

        # 진료비 상세 테이블
        draw.text((50, 240), "[ 진료비 세부내역 ]", fill="black", font=self._get_font(12))

        amounts = {
            "진찰료": random.randint(10, 30) * 1000,
            "검사료": random.randint(30, 100) * 1000,
            "영상진단료": random.randint(50, 200) * 1000,
            "투약료": random.randint(10, 50) * 1000,
            "주사료": random.randint(5, 30) * 1000,
            "처치료": random.randint(20, 80) * 1000,
        }
        total = sum(amounts.values())
        insurance_cover = int(total * 0.7)
        patient_pay = total - insurance_cover

        headers = ["항  목", "금  액", "급여", "비급여"]
        data = [[item, f"{amount:,}원", f"{int(amount*0.7):,}원", f"{int(amount*0.3):,}원"]
                for item, amount in amounts.items()]
        self._draw_table(draw, 50, 270, len(amounts) + 1, 4, cell_w=175, cell_h=32, headers=headers, data=data)
        annotations["words"].extend(list(amounts.keys()))

        # 합계 영역
        y_summary = 270 + (len(amounts) + 1) * 32 + 20
        draw.rectangle([50, y_summary, width - 50, y_summary + 120], fill="#f5f5f5", outline="black")

        summary_font = self._get_font(14)
        draw.text((70, y_summary + 10), f"총 진료비: {total:,}원", fill="black", font=summary_font)
        draw.text((70, y_summary + 40), f"보험자부담금: {insurance_cover:,}원", fill="black", font=summary_font)
        draw.text((70, y_summary + 70), f"환자부담금: {patient_pay:,}원", fill="red", font=self._get_font(16))
        annotations["words"].extend(["총진료비", "보험자부담금", "환자부담금"])

        # 수납 정보
        draw.text((70, y_summary + 100), f"수납액: {patient_pay:,}원  |  미수액: 0원",
                 fill="black", font=self._get_font(12))

        # 도장
        self._draw_stamp(draw, width - 120, y_summary + 50, 45, "수납")

        # 하단
        draw.line([(50, height - 120), (width - 50, height - 120)], fill="gray", width=1)
        draw.text((50, height - 100), hospital, fill="black", font=field_font)
        draw.text((50, height - 75), f"전화: 02-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
                 fill="gray", font=self._get_font(11))
        draw.text((50, height - 55), "※ 본 영수증은 소득공제용으로 사용 가능합니다.",
                 fill="gray", font=self._get_font(10))

        return self._add_noise(image), annotations

    def generate_prescription(self, index: int) -> Tuple[Image.Image, dict]:
        """처방전 생성 - 약품 테이블, QR코드, 의사서명"""
        width, height = 800, 1100
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        annotations = {"boxes": [], "words": [], "label": "처방전"}

        # 상단 병원 정보 바
        draw.rectangle([0, 0, width, 70], fill="#4CAF50")
        hospital = random.choice(self.HOSPITAL_NAMES)
        draw.text((50, 20), hospital, fill="white", font=self._get_font(20))
        draw.text((50, 48), f"Tel: 02-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
                 fill="white", font=self._get_font(11))
        annotations["words"].append(hospital)

        # 제목
        title_font = self._get_font(28)
        draw.text((width // 2 - 60, 90), "처 방 전", fill="black", font=title_font)
        annotations["words"].append("처방전")

        # QR코드 (우상단)
        self._draw_qrcode(draw, width - 110, 85, 70)

        # 환자 정보
        field_font = self._get_font(13)
        patient = random.choice(self.PATIENT_NAMES)

        draw.rectangle([50, 170, width - 50, 250], outline="black", width=1)
        draw.text((60, 180), f"환자성명: {patient}", fill="black", font=field_font)
        draw.text((350, 180), f"주민번호: {random.randint(700000, 999999)}-*******", fill="black", font=field_font)
        draw.text((60, 215), f"의료보험: {random.choice(['건강보험', '의료급여', '자동차보험'])}", fill="black", font=field_font)
        draw.text((350, 215), f"처방일: 2024.{random.randint(1,12):02d}.{random.randint(1,28):02d}", fill="black", font=field_font)
        annotations["words"].extend([patient, "환자성명", "주민번호", "처방일"])

        # 처방 의약품 테이블
        draw.text((50, 270), "[ 처방 의약품 ]", fill="black", font=self._get_font(14))

        headers = ["약품명", "용량", "횟수", "일수"]
        num_meds = random.randint(3, 5)
        data = []
        for _ in range(num_meds):
            med = random.choice(self.MEDICINE_NAMES)
            data.append([
                med,
                f"{random.choice([1, 2])}정",
                f"{random.choice([2, 3])}회",
                f"{random.choice([3, 5, 7])}일"
            ])

        self._draw_table(draw, 50, 300, num_meds + 1, 4, cell_w=175, cell_h=40, headers=headers, data=data)

        for row in data:
            annotations["words"].append(row[0])
        annotations["words"].extend(["약품명", "용량", "횟수", "일수"])

        # 조제 정보
        table_bottom = 300 + (num_meds + 1) * 40 + 20
        draw.rectangle([50, table_bottom, width - 50, table_bottom + 60], fill="#f9f9f9", outline="black")
        draw.text((60, table_bottom + 10), f"조제시 참고사항: {random.choice(['식후 30분', '취침 전', '공복 시'])} 복용",
                 fill="black", font=field_font)
        draw.text((60, table_bottom + 35), f"사용기간: {random.randint(3, 7)}일 이내 조제", fill="black", font=field_font)

        # 의사 정보
        doctor_y = table_bottom + 100
        draw.text((50, doctor_y), "처방의사", fill="black", font=self._get_font(12))
        draw.rectangle([50, doctor_y + 25, 350, doctor_y + 80], outline="black", width=1)
        draw.text((60, doctor_y + 35), "면허번호: 의사 제 12345호", fill="black", font=field_font)
        draw.text((60, doctor_y + 55), "성    명: 김의사", fill="black", font=field_font)

        # 서명
        self._draw_signature(draw, 250, doctor_y + 50, 80)

        # 도장
        self._draw_stamp(draw, 400, doctor_y + 55, 40, "인")

        # 조제약국 정보 (빈칸)
        draw.text((450, doctor_y), "조제약국", fill="black", font=self._get_font(12))
        draw.rectangle([450, doctor_y + 25, width - 50, doctor_y + 80], outline="black", width=1)
        draw.text((460, doctor_y + 45), "(약국 기재란)", fill="gray", font=self._get_font(11))

        # 하단 안내
        draw.line([(50, height - 100), (width - 50, height - 100)], fill="gray", width=1)
        small_font = self._get_font(10)
        draw.text((50, height - 85), "※ 이 처방전은 발행일로부터 3일간 유효합니다.", fill="gray", font=small_font)
        draw.text((50, height - 65), "※ 의약품 부작용 발생 시 의사 또는 약사에게 즉시 알리십시오.", fill="gray", font=small_font)

        return self._add_noise(image), annotations

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """이미지에 노이즈 추가"""
        img_array = np.array(image)
        noise = np.random.normal(0, 2, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def generate_document(self, doc_type: str, index: int) -> Tuple[Image.Image, dict]:
        """문서 유형에 따라 적절한 생성 함수 호출"""
        generators = {
            "진단서": self.generate_diagnosis,
            "소견서": self.generate_opinion,
            "보험금청구서": self.generate_insurance_claim,
            "입퇴원확인서": self.generate_admission_discharge,
            "의료비영수증": self.generate_medical_receipt,
            "처방전": self.generate_prescription,
        }
        return generators[doc_type](index)

    def generate_dataset(self, samples_per_class: int = 10, train_ratio: float = 0.8) -> dict:
        """전체 데이터셋 생성"""
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"

        for subdir in [train_dir / "images", train_dir / "annotations",
                       test_dir / "images", test_dir / "annotations"]:
            subdir.mkdir(parents=True, exist_ok=True)

        doc_types = ["진단서", "소견서", "보험금청구서", "입퇴원확인서", "의료비영수증", "처방전"]
        stats = {"train": {}, "test": {}}
        labels = {"train": {}, "test": {}}

        for doc_type in doc_types:
            n_train = int(samples_per_class * train_ratio)
            n_test = samples_per_class - n_train
            stats["train"][doc_type] = n_train
            stats["test"][doc_type] = n_test

            for i in range(n_train):
                image, annotations = self.generate_document(doc_type, i)
                filename = f"{doc_type}_{i:03d}"
                image.save(train_dir / "images" / f"{filename}.jpg", quality=95)
                annotations["image"] = f"{filename}.jpg"
                with open(train_dir / "annotations" / f"{filename}.json", "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)
                labels["train"][f"{filename}.jpg"] = doc_type

            for i in range(n_test):
                image, annotations = self.generate_document(doc_type, i + n_train)
                filename = f"{doc_type}_{i + n_train:03d}"
                image.save(test_dir / "images" / f"{filename}.jpg", quality=95)
                annotations["image"] = f"{filename}.jpg"
                with open(test_dir / "annotations" / f"{filename}.json", "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)
                labels["test"][f"{filename}.jpg"] = doc_type

        for split, split_labels in labels.items():
            split_dir = train_dir if split == "train" else test_dir
            with open(split_dir / "labels.tsv", "w", encoding="utf-8") as f:
                for filename, label in split_labels.items():
                    f.write(f"{filename}\t{label}\n")

        print(f"데이터셋 생성 완료:")
        print(f"  - 학습 데이터: {sum(stats['train'].values())}개")
        print(f"  - 테스트 데이터: {sum(stats['test'].values())}개")
        print(f"  - 저장 위치: {self.output_dir}")

        return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="샘플 문서 데이터 생성")
    parser.add_argument("--output-dir", default="data/sample", help="출력 디렉토리")
    parser.add_argument("--samples-per-class", type=int, default=10, help="클래스당 샘플 수")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="학습 데이터 비율")
    args = parser.parse_args()

    generator = SampleDocumentGenerator(args.output_dir)
    stats = generator.generate_dataset(samples_per_class=args.samples_per_class, train_ratio=args.train_ratio)

    print("\n클래스별 분포:")
    for split in ["train", "test"]:
        print(f"\n{split.upper()}:")
        for doc_type, count in stats[split].items():
            print(f"  - {doc_type}: {count}개")


if __name__ == "__main__":
    main()
