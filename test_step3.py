"""
test_step3.py —— 第3步端到端测试：批判性审查Agent
================================
运行方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  python test_step3.py

流程: 读取第1步元数据 + 第2步信息卡片 -> 程序化引用对齐
      -> LLM语义核验 -> 幻觉率量化报告

复用说明: 直接读取 papers_meta.json 和 paper_cards.json。
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PaperMeta, PaperCard
from agents import reviewer
import llm_client
import config

# ---------------------------------------------------------------
# 0. 加载第1、2步的产出
# ---------------------------------------------------------------
print("=" * 60)
print("第3步测试：批判性审查流水线")
print("=" * 60)

meta_path = os.path.join(config.DATA_DIR, "papers_meta.json")
cards_path = os.path.join(config.DATA_DIR, "paper_cards.json")
for path in (meta_path, cards_path):
    if not os.path.exists(path):
        print(f"✗ 未找到 {os.path.basename(path)}，请先运行前面的步骤")
        sys.exit(1)

with open(meta_path, "r", encoding="utf-8") as f:
    papers = [PaperMeta(**p) for p in json.load(f)]
with open(cards_path, "r", encoding="utf-8") as f:
    cards = [PaperCard(**c) for c in json.load(f)]
print(f"\n[测试] 加载 {len(papers)} 篇论文, {len(cards)} 张信息卡片\n")

# ---------------------------------------------------------------
# 1. 单元测试: 引用对齐函数
# ---------------------------------------------------------------
print("[测试] 单元测试: 引用对齐函数...")
sample_norm = reviewer._normalize(
    "The spiking neural network achieves 95.2 percent accuracy "
    "on the CIFAR-10 dataset with three time steps."
)
# 用例1: 忠实引用 -> 应对齐成功
assert reviewer.quote_alignment(
    "achieves 95.2 percent accuracy on the CIFAR-10 dataset", sample_norm
), "忠实引用被判为不匹配，阈值过严"
# 用例2: 编造内容 -> 应对齐失败
assert not reviewer.quote_alignment(
    "we prove that SNNs are biologically plausible in all cases", sample_norm
), "编造引用竟然对齐成功，检查匹配逻辑"
print("  ✓ 对齐函数工作正常（忠实引用通过，编造内容被拒）\n")

# ---------------------------------------------------------------
# 2. 端到端: 两级核验流水线
# ---------------------------------------------------------------
print("[测试] 开始两级核验（程序对齐 + LLM语义审查）...\n")
results = reviewer.run(papers, cards)

# ---------------------------------------------------------------
# 3. 幻觉率量化报告
# ---------------------------------------------------------------
report = reviewer.generate_report(results)

# ---------------------------------------------------------------
# 4. 结果检查
# ---------------------------------------------------------------
assert len(results) == sum(len(c.claims) for c in cards), "核验条数与声明总数不符"

# 展示被判为幻觉的声明明细（人工核查用，也是报告的分析素材）
print("\n[测试] 被判为幻觉/可疑的声明明细:")
has_bad = False
for r in results:
    is_halluc = (not r.quote_alignment) or r.verdict.verdict != "supported"
    if is_halluc:
        has_bad = True
        print(f"  [{r.arxiv_id} #{r.claim_index}] 对齐={'Y' if r.quote_alignment else 'N'}")
        print(f"    声明: {r.claim_content[:70]}...")
        print(f"    判定: {r.verdict.verdict} (置信度{r.verdict.confidence})")
        print(f"    理由: {r.verdict.reason[:70]}")
if not has_bad:
    print("  （无——所有声明均通过核验）")

llm_client.print_usage()
print("\n🎉 第3步测试通过！审查Agent完成两级核验与幻觉率量化。")
print(f"   产出位置: {config.DATA_DIR}\\review_results.json")
