"""
test_step2.py —— 第2步端到端测试：提取Agent
================================
运行方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  python test_step2.py

流程: 读取第1步产出的元数据 -> 解析PDF全文 -> LLM抽取信息卡片
      -> 抽查引用真实性（程序化预检，为第3步审查Agent铺垫）

复用说明: 直接读取 papers_meta.json，不需要重新检索下载。
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PaperMeta, PaperCard
from agents import extractor
import llm_client
import config

# ---------------------------------------------------------------
# 1. 加载第1步的产出（元数据 + 本地PDF）
# ---------------------------------------------------------------
print("=" * 60)
print("第2步测试：信息提取流水线")
print("=" * 60)

meta_path = os.path.join(config.DATA_DIR, "papers_meta.json")
if not os.path.exists(meta_path):
    print("✗ 未找到 papers_meta.json，请先运行 test_step1.py")
    sys.exit(1)

with open(meta_path, "r", encoding="utf-8") as f:
    papers = [PaperMeta(**p) for p in json.load(f)]
print(f"\n[测试] 加载 {len(papers)} 篇论文元数据\n")

# ---------------------------------------------------------------
# 2. 模块级单元测试: pdf_to_text
# ---------------------------------------------------------------
print("[测试] 单元测试: PDF解析...")
sample_text = extractor.pdf_to_text(papers[0].local_path)
assert len(sample_text) > 5000, f"PDF解析结果过短({len(sample_text)}字符)，可能解析失败"
assert "spiking" in sample_text.lower() or "neural" in sample_text.lower(), "解析内容疑似不对"
print(f"  ✓ 解析成功: {papers[0].arxiv_id}.pdf -> {len(sample_text)} 字符\n")

# ---------------------------------------------------------------
# 3. 端到端: 批量提取信息卡片
# ---------------------------------------------------------------
print("[测试] 开始批量提取（每篇论文约需1-2分钟，请耐心等待）...\n")
cards = extractor.run(papers)

# ---------------------------------------------------------------
# 4. 结果检查
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

assert len(cards) >= 3, "成功提取的论文太少"

total_claims, total_quotes = 0, 0
for card in cards:
    n_c = len(card.claims)
    n_q = sum(len(c.quotes) for c in card.claims)
    total_claims += n_c
    total_quotes += n_q
    types = [c.claim_type for c in card.claims]
    print(f"  {card.arxiv_id}: {n_c}条声明({n_q}条引用) "
          f"[method:{types.count('method')} result:{types.count('result')} "
          f"limitation:{types.count('limitation')}]")
    print(f"    标签: {card.method_category}")
    # 展示一条result类声明示例
    for c in card.claims:
        if c.claim_type == "result":
            print(f"    示例result: {c.content[:60]}...")
            print(f"      原文引用: \"{c.quotes[0].text[:60]}...\" ({c.quotes[0].section})")
            break

print(f"\n  合计: {len(cards)}篇论文, {total_claims}条声明, {total_quotes}条原文引用")

# ---------------------------------------------------------------
# 5. 引用真实性预检（第3步审查Agent的铺垫，这里只做简单包含检查）
# ---------------------------------------------------------------
print("\n[测试] 引用真实性预检（程序化粗检）...")
fake_count = 0
for card in cards:
    full_text = extractor.pdf_to_text(
        next(p.local_path for p in papers if p.arxiv_id == card.arxiv_id))
    norm_text = " ".join(full_text.split()).lower()  # 归一化空白+大小写
    for c in card.claims:
        for q in c.quotes:
            # 取引用的前80字符做归一化包含检查（粗检）
            frag = " ".join(q.text.split()).lower()[:80]
            if frag and frag not in norm_text:
                fake_count += 1
                print(f"  ⚠ 疑似不匹配: [{card.arxiv_id}] \"{q.text[:50]}...\"")

print(f"  预检结果: {total_quotes}条引用中 {fake_count} 条疑似不匹配 "
      f"({(total_quotes - fake_count) / total_quotes:.0%}通过率)")

llm_client.print_usage()
print("\n🎉 第2步测试通过！信息卡片已生成，可以进入第3步（审查Agent）。")
print(f"   产出位置: {config.DATA_DIR}\\paper_cards.json")
