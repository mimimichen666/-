"""
test_evidence.py —— 方案一"证据链穿透"核心引擎测试
================================
验证三件事:
  1. locate_quote三级匹配的命中率（目标>85%）
  2. 高亮渲染成功率
  3. evidence_index.json索引完整性

运行: conda run -n brain python -u test_evidence.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import evidence_locator

cards = json.load(open("data/paper_cards.json", encoding="utf-8"))
try:
    reviews = json.load(open("data/review_results.json", encoding="utf-8"))
except FileNotFoundError:
    reviews = None

# 逐条测试定位（详细模式，看每条的匹配级别）
print("=" * 60)
print("逐条定位测试:")
total = hit = 0
for card in cards:
    arxiv_id = card["arxiv_id"]
    pdf_path = os.path.join("papers", f"{arxiv_id.replace('/', '_')}.pdf")
    if not os.path.exists(pdf_path):
        print(f"  [无PDF] {arxiv_id}")
        continue
    for idx, claim in enumerate(card.get("claims", [])):
        total += 1
        located = None
        for q in claim.get("quotes", []):
            located = evidence_locator.locate_quote(
                pdf_path, q.get("text", ""))
            if located:
                break
        if located:
            hit += 1
            page, matched = located
            print(f"  ✓ {arxiv_id}#{idx} [{claim['claim_type']:10s}] "
                  f"-> 第{page + 1}页 (匹配{len(matched)}字符)")
        else:
            quote_preview = (claim["quotes"][0]["text"]
                             if claim.get("quotes") else "")[:60]
            print(f"  ✗ {arxiv_id}#{idx} [{claim['claim_type']:10s}] "
                  f"未定位 | quote: {quote_preview}...")

print(f"\n定位命中率: {hit}/{total} = {hit * 100 // max(total, 1)}%")
print("=" * 60)

# 完整流程: 建索引(含高亮渲染)
print("\n批量生成证据索引:")
index = evidence_locator.build_evidence_index(cards, reviews)

# 索引统计
located_n = sum(1 for v in index.values() if v.get("located"))
print(f"\n索引条目: {len(index)}, 定位成功: {located_n}")
imgs = [v["image"] for v in index.values() if v.get("located")]
print(f"证据图片已生成: {len(imgs)}张")
for img in imgs[:5]:
    p = os.path.join("evidence", img)
    size = os.path.getsize(p) // 1024
    print(f"  - {img} ({size}KB)")
