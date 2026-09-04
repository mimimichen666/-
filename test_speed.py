"""
test_speed.py —— 检索速度计时测试（优化效果验证）
================================
只跑检索阶段（检索+粗筛+下载），不跑提取/审查，用于测量优化后的耗时。

运行: conda run -n brain python -u test_speed.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SearchPlan
from agents import searcher

# 模拟规划Agent产出的3组检索词（优化点1: 从5组收敛到3组）
plan = SearchPlan(
    topic="transformer efficiency",
    keywords=[
        "efficient transformer models",
        "model compression transformers",
        "knowledge distillation transformer",
    ],
    inclusion_criteria="近5年、与主题相关、有实验结果",
    target_count=5,
)

t0 = time.time()
papers = searcher.search(plan, results_per_query=8)
t1 = time.time()
print(f">>> [计时] 检索阶段: {t1 - t0:.1f}秒, 获得{len(papers)}篇候选")

papers = searcher.filter_papers(plan, papers)
t2 = time.time()
print(f">>> [计时] LLM粗筛: {t2 - t1:.1f}秒, 保留{len(papers)}篇")

downloaded = searcher.download(papers[:4])
t3 = time.time()
print(f">>> [计时] PDF并行下载: {t3 - t2:.1f}秒, 成功{len(downloaded)}/{min(4, len(papers))}篇")
print(f">>> [计时] 检索Agent总耗时: {t3 - t0:.1f}秒")
