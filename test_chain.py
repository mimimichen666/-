"""
test_chain.py —— 引用链挖掘 + 四路综合排序 验证
================================
验证目标:
  1. 引用链挖掘能否召回关键词检索不到的奠基经典
     （SpikeProp/tempotron/STDP等老论文的摘要不含现代检索词）
  2. 共被引次数(chain_hits)是否正确填充
  3. 经典免检通道(chain_hits>=3跳过LLM粗筛)是否生效
  4. 四路综合排序后经典论文的最终名次

运行: conda run -n brain python -u test_chain.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SearchPlan
from agents import searcher

# SNN训练主题（与test_rrf.py同一计划，保证结果可对比）
# classic_titles模拟规划Agent的LLM领域先验输出（真实流程由planner生成）
plan = SearchPlan(
    topic="脉冲神经网络的高效训练方法",
    keywords=[
        "spiking neural network training backpropagation",
        "surrogate gradient spiking neural network",
        "SNN neuromorphic learning algorithm",
    ],
    classic_titles=[
        "SpikeProp: backpropagation for networks of spiking neurons",
        "The tempotron: a neuron that learns spike timing-based decisions",
        "ReSuMe: new supervised learning method for spiking neural networks",
        "Unsupervised learning of digit recognition using spike-timing-dependent plasticity",
        "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning",
        "SLAYER: Spike Layer Error Reassignment in Time",
    ],
    inclusion_criteria="与SNN训练方法直接相关、有实验结果",
    target_count=10,
)

# ---- 第1步: 检索 + 引用链挖掘 ----
candidates = searcher.search(plan, results_per_query=25)

print(f"\n{'=' * 70}")
print(f"候选总数: {len(candidates)}篇（含引用链挖出的经典）")
print("=" * 70)

# 引用链挖出的论文单独展示（chain_hits>0）
chained = [p for p in candidates if p.chain_hits > 0]
print(f"\n引用链信号论文共{len(chained)}篇，按共被引次数排序:")
for p in sorted(chained, key=lambda p: -p.chain_hits)[:15]:
    print(f"  [共被引{p.chain_hits} 被引{p.cited_by:>5}] ({p.year}) "
          f"{p.title[:58]}")

# ---- 第2步: 经典论文召回检查 ----
# 关键词检索的"盲区论文"——标题不含任何检索词，只有引用链能召回
print(f"\n{'=' * 70}")
print("奠基经典召回检查（这些论文的标题/摘要不含现代检索词）:")
print("=" * 70)
blind_classics = {
    "SpikeProp": "SpikeProp（SNN反向传播开山之作1999）",
    "tempotron": "Tempotron（脉冲神经元监督学习2005）",
    "ReSuMe": "ReSuMe（远程监督学习规则）",
    "spike-timing-dependent": "STDP学习规则经典",
    "Unsupervised learning of digit": "Diehl&Cook STDP识别经典",
    "Loihi": "Loihi神经形态芯片",
    "slayer": "SLAYER训练算法",
    "Surrogate": "代理梯度类方法",
}
for key, desc in blind_classics.items():
    hits = [p for p in candidates if key.lower() in p.title.lower()]
    if hits:
        best = max(hits, key=lambda p: p.chain_hits)
        src = "引用链" if best.chain_hits > 0 else "关键词"
        print(f"  ✓ {desc}: {src}召回 | {best.title[:48]} "
              f"(被引{best.cited_by}, 共被引{best.chain_hits})")
    else:
        print(f"  ✗ {desc}: 未召回")

# ---- 第3步: LLM粗筛（含经典免检）+ 综合排序 ----
print(f"\n{'=' * 70}")
print("LLM粗筛（观察经典免检通道）+ 四路综合排序:")
print("=" * 70)
filtered = searcher.filter_papers(plan, candidates, threshold=4.0)
ranked = searcher._composite_sort(filtered)
print(f"\n最终排序（用户实际看到的顺序，满分5）:")
for i, p in enumerate(ranked[:15], 1):
    rel = p.relevance_score if p.relevance_score is not None else "免检"
    print(f"{i:2d}. [final={p.final_score} 相关={rel} 被引={p.cited_by:>4} "
          f"链={p.chain_hits}] {p.title[:52]}")
