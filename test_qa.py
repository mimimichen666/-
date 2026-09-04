"""
test_qa.py —— 方案二"可信问答"测试
================================
验证:
  1. 声明池构建正确（只含核验通过的声明）
  2. 可答组: 声明池内的问题正常回答且带引用
  3. 拒答组: 知识库外的问题正确拒答
  4. 一个真实问答的完整输出展示

运行: conda run -n brain python -u test_qa.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import qa_agent
import llm_client

# ---- 1. 声明池 ----
pool = qa_agent.build_claim_pool()
print(f"声明池: {len(pool)}条已核验声明")
for c in pool[:5]:
    print(f"  [#{c['num']}] ({c['type']:10s}) {c['content'][:55]}")
print("  ...")

# ---- 2. 拒答测试 ----
print("\n" + "=" * 60)
result = qa_agent.run_refusal_test()

# ---- 3. 真实问答演示 ----
print("\n" + "=" * 60)
print("真实问答演示:")
demo_q = "这些论文各自的核心方法是什么？"
print(f"\n用户: {demo_q}")
ans, used_pool = qa_agent.answer_question(demo_q)
print(f"can_answer: {ans.can_answer}")
print(f"使用声明: {ans.used_claims}")
print(f"回答:\n{ans.answer[:500]}")

llm_client.print_usage()
