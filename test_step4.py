"""
test_step4.py —— 第4步端到端测试：综合Agent
================================
运行方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  python test_step4.py

流程: 读取前三步产出 -> 综述表格 -> 引用图谱(S2真实引用,失败自动降级二部图)
      -> 最终Markdown报告 -> 检查全部产出
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import synthesizer
import llm_client

TOPIC = "脉冲神经网络的高效训练方法"   # 与第1步检索时一致

# ---------------------------------------------------------------
# 运行完整流水线
# ---------------------------------------------------------------
print("=" * 60)
print("第4步测试：综合Agent流水线")
print("=" * 60 + "\n")

outputs = synthesizer.run(TOPIC)

# ---------------------------------------------------------------
# 产出检查
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("产出检查")
print("=" * 60)

# 1. 表格
assert os.path.exists(outputs["table"]), "综述表格未生成"
table = open(outputs["table"], encoding="utf-8").read()
assert "| 论文 |" in table, "表格格式异常"
assert "方法类别" in table, "表格缺少分类分组"
print(f"  ✓ 综述表格: {len(table)} 字符, 含{table.count('## 方法类别')}个方法类别")

# 2. 图谱
assert os.path.exists(outputs["graph"]), "引用图谱未生成"
html = open(outputs["graph"], encoding="utf-8").read()
assert "nodes" in html and len(html) > 5000, "图谱HTML内容异常"
print(f"  ✓ 引用图谱: 交互式HTML ({len(html) // 1024} KB)")

# 3. 报告
assert os.path.exists(outputs["report"]), "最终报告未生成"
report = open(outputs["report"], encoding="utf-8").read()
for section in ("方法分类与发展脉络", "论文对比总表", "引用图谱",
                "审查统计", "附录B"):
    assert section in report, f"报告缺少章节: {section}"
assert "arXiv:" in report, "报告缺少论文编号引用标注"
assert "幻觉率" in report, "报告缺少幻觉率统计"
print(f"  ✓ 最终报告: {len(report)} 字符, 五大章节齐全")

# 展示报告的核心统计段落
import re
m = re.search(r"## 四、审查统计.*?(?=\n## )", report, re.S)
if m:
    print("\n" + m.group(0).strip())

llm_client.print_usage()
print("\n🎉 第4步测试通过！全部四大Agent流水线已跑通。")
print("产出位置: d:\\trae项目\\literature_agent\\output\\")
print("  - review_table.md    综述对比表格")
print("  - citation_graph.html 交互式引用图谱(浏览器打开)")
print("  - final_report.md    最终综述报告")
