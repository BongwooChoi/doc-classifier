#!/usr/bin/env python3
"""API 서버 실행 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from api.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("문서 분류 API 서버 시작")
    print("=" * 60)
    print("\n접속 URL: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("\nCtrl+C로 종료\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
