"""
test_rrf.py —— 检索质量改进验证（多路检索+RRF融合）
================================
对比验证:
  1. 双路检索返回的候选质量（经典论文是否进入候选池）
  2. RRF融合排序 vs 旧版单一被引排序 的头部论文差异
  3. SNN领域经典论文检查表（人工已知的经典工作是否被召回）

运行: conda run -n brain python -u test_rrf.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SearchPlan
from agents import searcher

# SNN主题的固定检索计划（含多种角度的检索词——RRF多关键词加成的测试场景）
plan = SearchPlan(
    topic="脉冲神经网络的高效训练方法",
    keywords=[
        "spiking neural network training backpropagation",
        "surrogate gradient spiking neural network",
        "SNN neuromorphic learning algorithm",
    ],
    inclusion_criteria="与SNN训练方法直接相关、有实验结果",
    target_count=10,
)

# 只测检索+融合段（不做LLM粗筛和下载，聚焦排序质量）
candidates = searcher.search(plan, results_per_query=25)

print(f"\n{'=' * 70}")
print(f"候选总数: {len(candidates)}篇, 按RRF融合分排序的前20名:")
print("=" * 70)
for i, p in enumerate(candidates[:20]):
    # 标记是否被多个关键词命中（rrf_score自然体现）
    print(f"{i + 1:2d}. [RRF={p.rrf_score:.4f} 被引={p.cited_by:>5}] "
          f"({p.year}) {p.title[:58]}")

# ---- SNN经典论文检查表 ----
# 人工领域知识给出的经典工作关键词（标题特征）
classics = {
    "SpikeProp": "SpikeProp（SNN反向传播开山之作）",
    "Surrogate": "代理梯度类方法",
    "SpikingJelly": "SpikingJelly框架",
    "SEW": "SEW-ResNet",
    "temporal coding": "时间编码",
    "STDP": "STDP学习规则",
    "Loihi": "Loihi神经形态芯片",
    "slayer": "SLAYER训练算法",
}
print(f"\n{'=' * 70}")
print("经典论文召回检查（标题关键词匹配）:")
print("=" * 70)
for key, desc in classics.items():
    hits = [p for p in candidates if key.lower() in p.title.lower()]
    if hits:
        best = hits[0]
        rank = candidates.index(best) + 1
        print(f"  ✓ {desc}: 排名{rank} | {best.title[:50]} "
              f"(被引{best.cited_by})")
    else:
        print(f"  ✗ {desc}: 未召回")

# ---- 排序对比: 单一被引 vs RRF融合 ----
print(f"\n{'=' * 70}")
print("排序策略对比（前10名差异）:")
print("=" * 70)
by_cited = sorted(candidates, key=lambda p: p.cited_by, reverse=True)
rrf_top10 = {p.arxiv_id for p in candidates[:10]}
cited_top10 = {p.arxiv_id for p in by_cited[:10]}
overlap = rrf_top10 & cited_top10
print(f"  RRF前10 ∩ 纯被引前10 = {len(overlap)}篇")

# ---- 端到端验证: LLM粗筛后的最终排序(用户实际看到的顺序) ----
print(f"\n{'=' * 70}")
print("LLM粗筛 + 综合排序后的最终顺序 (用户实际看到的):")
print("=" * 70)
filtered = searcher.filter_papers(plan, candidates, threshold=4.0)
import math
if filtered:
    max_rrf = max(p.rrf_score for p in filtered) or 1.0
    max_cit = max(p.cited_by for p in filtered)
    scored = []
    for p in filtered:
        impact = (math.log1p(p.cited_by) / math.log1p(max_cit)
                  if max_cit > 0 else 0.0)
        final = (0.6 * (p.relevance_score or 0)
                 + 0.25 * 5.0 * (p.rrf_score / max_rrf)
                 + 0.15 * 5.0 * impact)
        scored.append((final, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    for i, (final, p) in enumerate(scored[:12]):
        print(f"{i + 1:2d}. [final={final:.3f} LLM={p.relevance_score} "
              f"被引={p.cited_by:>3}] {p.title[:52]}")
