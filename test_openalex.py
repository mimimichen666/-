"""
test_openalex.py —— OpenAlex主数据源回归测试
================================
验证三件事:
  1. OpenAlex检索被正确选为主源（看日志"数据源选定: OpenAlex"）
  2. "近5年"年份过滤生效（返回论文年份都>=2022）
  3. 下载的文件真的是PDF（魔数校验，非PDF自动跳过）

运行: conda run -n brain python -u test_openalex.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SearchPlan
from agents import searcher

# 与此前arXiv基线测试相同的SNN主题（便于对比）
plan = SearchPlan(
    topic="spiking neural network efficient training",
    keywords=[
        "spiking neural network training",
        "neuromorphic computing learning",
        "SNN backpropagation surrogate gradient",
    ],
    inclusion_criteria="近5年、与SNN高效训练相关、有实验结果",
    target_count=5,
)

t0 = time.time()
papers = searcher.search(plan, results_per_query=8)
t1 = time.time()
print(f"\n>>> [计时] 检索阶段: {t1 - t0:.1f}秒, 获得{len(papers)}篇候选")

# 验证年份过滤（近5年 → 2022年起）
years = [p.year for p in papers]
bad_years = [y for y in years if y < 2022]
print(f">>> [验证] 年份分布: min={min(years) if years else '-'}, "
      f"max={max(years) if years else '-'}, 违规(<2022)={len(bad_years)}条")

for p in papers[:10]:
    print(f"    - [{p.year}] {p.title[:58]} (id={p.arxiv_id})")

# 下载前4篇验证PDF魔数校验
downloaded = searcher.download(papers[:4])
t2 = time.time()
print(f"\n>>> [计时] 下载阶段: {t2 - t1:.1f}秒, 成功{len(downloaded)}/4")

# 终极验证: PyMuPDF能否解析（下游extractor的硬要求）
import fitz
ok = 0
for p in downloaded:
    try:
        doc = fitz.open(p.local_path)
        pages = len(doc)
        doc.close()
        ok += 1
        print(f"    [PDF解析OK] {os.path.basename(p.local_path)} ({pages}页)")
    except Exception as e:
        print(f"    [PDF解析失败] {p.local_path}: {e}")

print(f"\n>>> 总结: 检索{t1-t0:.1f}s + 下载{t2-t1:.1f}s, "
      f"PyMuPDF可解析 {ok}/{len(downloaded)}")
