"""
test_conflict.py —— 方案三"矛盾检测"测试
================================
验证:
  1. 预筛压缩比（全量对 -> 候选对）
  2. LLM判定的争议质量（contradict/tension/consistent区分度）
  3. conflicts.json落盘

运行: conda run -n brain python -u test_conflict.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import conflict_detector
import llm_client

results = conflict_detector.run()

print(f"\n========== 检测结果: {len(results)}处争议 ==========")
for c in results:
    icon = "🔴" if c["relation"] == "contradict" else "🟡"
    print(f"\n{icon} [{c['relation']}|严重度{c['severity']}] 主题: {c['topic']}")
    print(f"  A({c['a']['year']}, arXiv:{c['a']['arxiv_id']}): "
          f"{c['a']['content'][:55]}")
    print(f"  B({c['b']['year']}, arXiv:{c['b']['arxiv_id']}): "
          f"{c['b']['content'][:55]}")
    print(f"  分析: {c['explanation']}")

if not results:
    print("\n(未发现争议——当前声明池论文主题差异较大时属正常)")

llm_client.print_usage()
