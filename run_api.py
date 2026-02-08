#!/usr/bin/env python3
"""API 서버 실행 스크립트"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("문서 분류 API 서버 시작")
    print("=" * 60)
    print("\n접속 URL: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("\nCtrl+C로 종료\n")

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
