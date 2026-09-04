"""
test_e2e.py —— 端到端完整流水线验证（无Streamlit UI版本）
================================
复刻 app.py run_pipeline 的完整流程:
  规划 -> 检索(含引用链+经典注入) -> 提取 -> 审查 -> 综合
  -> 证据定位 -> 矛盾检测
验证新检索体系(四路信号召回+综合排序)在真实流水线中的效果。

运行: conda run -n brain python -u test_e2e.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (planner, searcher, extractor, reviewer, synthesizer,
                    evidence_locator, conflict_detector)
import config

TOPIC = "脉冲神经网络的高效训练方法"
TARGET_COUNT = 12

t0 = time.time()

# ---- 1. 规划Agent（真实LLM生成, 含classic_titles领域先验）----
print("=" * 70)
print("第1步 规划Agent")
print("=" * 70)
plan = planner.make_plan(TOPIC)
print(f"经典论文清单({len(plan.classic_titles)}篇):")
for t in plan.classic_titles:
    print(f"  - {t}")

# ---- 2. 检索Agent（四路信号: 关键词双路+引用链+LLM先验）----
print("\n" + "=" * 70)
print("第2步 检索Agent（关键词双路 + 引用链挖掘 + 经典注入）")
print("=" * 70)
papers = searcher.run(plan, results_per_query=8, top_k=TARGET_COUNT)
print(f"\n最终入库 {len(papers)} 篇（按final_score降序）:")
for i, p in enumerate(papers, 1):
    rel = p.relevance_score if p.relevance_score is not None else "免检"
    print(f"{i:2d}. [final={p.final_score} rel={rel} 被引={p.cited_by:>5} "
          f"链={p.chain_hits}] ({p.year}) {p.title[:50]}")

# ---- 3. 提取Agent ----
print("\n" + "=" * 70)
print("第3步 提取Agent")
print("=" * 70)
cards = extractor.run(papers)
print(f"提取 {len(cards)} 张卡片, "
      f"{sum(len(c.claims) for c in cards)} 条声明")

# ---- 4. 审查Agent（两级核验）----
print("\n" + "=" * 70)
print("第4步 审查Agent")
print("=" * 70)
results = reviewer.run(papers, cards)
report = reviewer.generate_report(results)
print(f"幻觉率: {report.hallucination_rate:.1%} | "
      f"原文支持: {report.supported}/{report.total_claims} | "
      f"不支持{report.unsupported} 矛盾{report.contradicted} "
      f"伪造引用{report.fake_quotes}")

# ---- 5. 综合Agent ----
print("\n" + "=" * 70)
print("第5步 综合Agent")
print("=" * 70)
outputs = synthesizer.run(TOPIC)
report_size = os.path.getsize(outputs["report"]) if os.path.exists(
    outputs["report"]) else 0
print(f"综述报告已生成: {outputs['report']} ({report_size}字节)")
print(f"综述表格: {outputs['table']}")
print(f"引用图谱: {outputs['graph']}")

# ---- 6. 证据定位 ----
print("\n" + "=" * 70)
print("第6步 证据定位（证据链穿透）")
print("=" * 70)
cards_raw = json.load(open(
    os.path.join(config.DATA_DIR, "paper_cards.json"), encoding="utf-8"))
reviews_raw = json.load(open(
    os.path.join(config.DATA_DIR, "review_results.json"), encoding="utf-8"))
ev_index = evidence_locator.build_evidence_index(cards_raw, reviews_raw)
located = sum(1 for v in ev_index.values() if v.get("located"))
print(f"证据定位: {located}/{len(ev_index)} 条声明可穿透到PDF原文")

# ---- 7. 矛盾检测 ----
print("\n" + "=" * 70)
print("第7步 矛盾检测（学术争议发现）")
print("=" * 70)
conflicts = conflict_detector.run()
n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
print(f"矛盾检测: 直接矛盾{n_con}处 + 张力{len(conflicts) - n_con}处")

# ---- 汇总 ----
elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"端到端流水线完成! 总耗时 {elapsed / 60:.1f} 分钟")
print(f"论文 {len(papers)} 篇 | 卡片 {len(cards)} 张 | "
      f"幻觉率 {report.hallucination_rate:.1%} | "
      f"证据穿透 {located} 条 | 争议 {len(conflicts)} 处")
print("=" * 70)

import llm_client
llm_client.print_usage()
