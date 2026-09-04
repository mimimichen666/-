"""
test_broad_topic.py —— 宽泛主题检索数量回归测试
================================
复现用户报告的问题: 搜索"计算机科学与技术"只得2篇。
验证修复: 检索量随目标放大(top_k*2/词) + 下载超额补位。

运行: conda run -n brain python -u test_broad_topic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SearchPlan
from agents import searcher
import llm_client

# 模拟规划Agent对"计算机科学与技术"这种宽泛主题的产出
# （宽泛主题的3组检索词会比较泛，粗筛淘汰率高，正好考验余量机制）
plan = SearchPlan(
    topic="computer science and technology",
    keywords=[
        "computer science",
        "computing technology survey",
        "computer systems",
    ],
    inclusion_criteria="近5年、与计算机科学与技术相关、有研究贡献",
    target_count=15,
)

papers = searcher.run(plan, results_per_query=8, top_k=15)

print(f"\n========== 最终结果: {len(papers)}/15 篇 ==========")
for p in papers:
    print(f"  [{p.year}] {p.title[:60]} (score={p.relevance_score})")

llm_client.print_usage()
