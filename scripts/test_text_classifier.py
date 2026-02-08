#!/usr/bin/env python3
"""텍스트 기반 분류기 테스트 (OCR 결과 활용)"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


class SimpleTextClassifier:
    """키워드 기반 단순 텍스트 분류기"""

    KEYWORDS = {
        "진단서": {
            "must_have": ["진단"],
            "boost": ["환자", "병명", "진단일", "치료기간", "진단함"],
            "weight": 1.0
        },
        "소견서": {
            "must_have": ["소견"],
            "boost": ["환자", "소견내용", "치료계획", "소견함"],
            "weight": 1.0
        },
        "보험금청구서": {
            "must_have": ["보험금", "청구"],
            "boost": ["피보험자", "증권번호", "청구금액", "사고일자"],
            "weight": 1.0
        },
        "입퇴원확인서": {
            "must_have": ["입퇴원", "입원", "퇴원"],
            "boost": ["입원일", "퇴원일", "병동", "확인서"],
            "weight": 1.0
        },
        "의료비영수증": {
            "must_have": ["영수증", "의료비", "진료비"],
            "boost": ["진찰료", "검사료", "처치료", "합계", "원"],
            "weight": 1.0
        },
        "처방전": {
            "must_have": ["처방"],
            "boost": ["약품명", "용량", "용법", "일수", "처방일"],
            "weight": 1.0
        }
    }

    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """텍스트 분류

        Args:
            text: 분류할 텍스트

        Returns:
            (예측 클래스, 신뢰도, 모든 점수)
        """
        text_lower = text.lower()

        scores = {}

        for doc_class, keywords in self.KEYWORDS.items():
            score = 0.0

            must_have_found = False
            for keyword in keywords["must_have"]:
                if keyword in text:
                    must_have_found = True
                    score += 2.0
                    break

            if must_have_found:
                for boost_keyword in keywords["boost"]:
                    if boost_keyword in text:
                        score += 0.5

                score *= keywords["weight"]

            scores[doc_class] = score

        total = sum(scores.values())
        if total > 0:
            for cls in scores:
                scores[cls] /= total

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predicted_class = sorted_scores[0][0]
        confidence = sorted_scores[0][1]

        return predicted_class, confidence, scores


def test_classifier():
    """분류기 테스트"""
    print("=" * 60)
    print("텍스트 기반 분류기 테스트")
    print("=" * 60)

    classifier = SimpleTextClassifier()

    test_dir = Path("data/sample/test/annotations")
    labels_file = Path("data/sample/test/labels.tsv")

    labels = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels[parts[0].replace(".jpg", "")] = parts[1]

    results = []

    for anno_file in sorted(test_dir.glob("*.json")):
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = " ".join(data.get("words", []))
        gt_label = data.get("label", labels.get(anno_file.stem, "unknown"))

        predicted, confidence, scores = classifier.classify(text)

        is_correct = predicted == gt_label
        results.append({
            "file": anno_file.stem,
            "ground_truth": gt_label,
            "predicted": predicted,
            "confidence": confidence,
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"\n{status} {anno_file.stem}")
        print(f"   텍스트: {text[:80]}...")
        print(f"   정답: {gt_label} | 예측: {predicted} ({confidence:.1%})")

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"결과 요약")
    print("=" * 60)
    print(f"\n정확도: {correct}/{total} = {accuracy:.1%}")

    class_results = {}
    for r in results:
        gt = r["ground_truth"]
        if gt not in class_results:
            class_results[gt] = {"correct": 0, "total": 0}
        class_results[gt]["total"] += 1
        if r["correct"]:
            class_results[gt]["correct"] += 1

    print("\n클래스별 정확도:")
    for cls, stats in sorted(class_results.items()):
        cls_acc = stats["correct"] / stats["total"]
        print(f"  - {cls}: {stats['correct']}/{stats['total']} = {cls_acc:.1%}")

    return accuracy


def test_train_data():
    """학습 데이터로도 테스트"""
    print("\n" + "=" * 60)
    print("학습 데이터 텍스트 분류 테스트")
    print("=" * 60)

    classifier = SimpleTextClassifier()
    train_dir = Path("data/sample/train/annotations")

    results = []
    for anno_file in sorted(train_dir.glob("*.json")):
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = " ".join(data.get("words", []))
        gt_label = data.get("label", "unknown")

        predicted, confidence, _ = classifier.classify(text)
        is_correct = predicted == gt_label
        results.append(is_correct)

    correct = sum(results)
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n학습 데이터 정확도: {correct}/{total} = {accuracy:.1%}")

    return accuracy


if __name__ == "__main__":
    test_accuracy = test_classifier()
    train_accuracy = test_train_data()

    print("\n" + "=" * 60)
    print("결론")
    print("=" * 60)
    print(f"\n✓ 텍스트 기반 분류기로 테스트 정확도 {test_accuracy:.1%} 달성")
    print(f"✓ 학습 데이터 정확도: {train_accuracy:.1%}")
    print("\n이 접근 방식은 LayoutLM 학습 데이터 준비에 활용할 수 있습니다.")
