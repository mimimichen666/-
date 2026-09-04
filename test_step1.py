"""
test_step1.py —— 第1步端到端测试：规划Agent + 检索Agent
================================
运行方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  python test_step1.py

流程: 输入中文主题 -> 规划Agent生成检索计划 -> arXiv检索
      -> LLM粗筛 -> 下载PDF -> 检查产出
"""

import os
import sys

# 把项目根目录加入sys.path（因为agents/下的模块要import根目录的文件）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import planner, searcher
import llm_client
import config

# ---------------------------------------------------------------
# 1. 规划Agent：主题 -> 检索计划
# ---------------------------------------------------------------
print("=" * 60)
print("第1步测试：科研文献检索流水线")
print("=" * 60)

TOPIC = "脉冲神经网络的高效训练方法"   # 测试主题（可以改成任何你想调研的主题）

print(f"\n[测试] 输入主题: {TOPIC}\n")
plan = planner.make_plan(TOPIC)

# ---------------------------------------------------------------
# 2. 检索Agent：计划 -> 论文PDF + 元数据JSON
# ---------------------------------------------------------------
print("\n[测试] 开始检索流水线（注意: arXiv限速，每步间隔3秒，请耐心等待）...\n")
papers = searcher.run(plan, results_per_query=8, top_k=5)

# ---------------------------------------------------------------
# 3. 结果检查
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)
print(f"最终获得论文: {len(papers)} 篇")
for i, p in enumerate(papers, 1):
    size = (os.path.getsize(p.local_path) / 1024) if p.local_path else 0
    print(f"  {i}. [{p.relevance_score}分] ({p.year}) {p.title[:60]}...")
    print(f"     arXiv: {p.arxiv_id} | PDF: {size:.0f}KB")

# 断言式检查（全部通过才算测试成功）
assert len(papers) >= 3, "过筛论文太少，检查检索词或阈值设置"
assert all(p.local_path for p in papers), "存在未下载成功的论文"
assert os.path.exists(os.path.join(config.DATA_DIR, "papers_meta.json")), "元数据JSON未生成"

llm_client.print_usage()
print("\n🎉 第1步测试通过！检索Agent流水线端到端跑通。")
print("   产出位置:")
print(f"   - PDF: {config.PAPER_DIR}")
print(f"   - 元数据: {config.DATA_DIR}\\papers_meta.json")
