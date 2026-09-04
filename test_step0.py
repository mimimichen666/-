"""
test_step0.py —— 第0步环境验证脚本
================================
运行方法（PowerShell）:
  conda activate brain
  cd d:\trae项目\literature_agent
  $env:LLM_API_KEY="你的API密钥"     # DeepSeek或智谱的key
  python test_step0.py

验证内容（按顺序）:
  1. 依赖库能否导入
  2. Pydantic数据结构定义是否正确
  3. LLM API是否能正常调用（需要API密钥）
  4. chat_json 强制结构化输出是否工作
"""

import sys

# ---------------------------------------------------------------
# 检查1：依赖库导入
# ---------------------------------------------------------------
print("-" * 50)
print("检查1：依赖库导入...")
try:
    import pymupdf       # PyMuPDF，PDF解析（新版模块名，旧名fitz已弃用）
    import arxiv         # arXiv检索
    import networkx      # 引用图谱
    import pyvis         # 图谱可视化
    import streamlit     # 前端
    import openai        # LLM调用
    import pydantic      # 数据结构
    print(f"  ✓ 全部导入成功")
    print(f"    PyMuPDF={pymupdf.__version__}, "
          f"openai={openai.__version__}, pydantic={pydantic.__version__}")
except ImportError as e:
    print(f"  ✗ 缺少依赖: {e}")
    print("  请运行: conda run -n brain pip install PyMuPDF arxiv pyvis streamlit")
    sys.exit(1)

# ---------------------------------------------------------------
# 检查2：Pydantic结构定义
# ---------------------------------------------------------------
print("-" * 50)
print("检查2：Pydantic数据结构...")
try:
    from models import SearchPlan, PaperCard, Quote, Claim, HallucinationReport

    # 手工构造一个最小样例，验证字段校验逻辑
    demo = SearchPlan(
        topic="脉冲神经网络",
        keywords=["spiking neural network", "SNN training"],
        inclusion_criteria="近5年、有实验结果",
        target_count=10,
    )
    print(f"  ✓ SearchPlan 构造成功: {demo.keywords}")

    # 验证幻觉率计算属性
    report = HallucinationReport(
        total_claims=10, supported=7, unsupported=2, contradicted=1, fake_quotes=1
    )
    print(f"  ✓ 幻觉率计算: {report.hallucination_rate:.0%} (预期 40%)")

    # 验证空quotes会被拒绝（防幻觉的强制约束）
    try:
        Claim(claim_type="method", content="测试声明", quotes=[])
        print("  ✗ 空quotes竟然通过了校验！min_length约束失效")
        sys.exit(1)
    except Exception:
        print("  ✓ 空quotes被正确拒绝（引用强制约束生效）")

except Exception as e:
    print(f"  ✗ models.py 结构错误: {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# 检查3：LLM API连通性（需要API密钥）
# ---------------------------------------------------------------
print("-" * 50)
print("检查3：LLM API 连通性...")
import config
if not config.API_KEY:
    print("  ⚠ 未检测到API密钥，跳过此检查")
    print("    设置方法: $env:LLM_API_KEY='sk-xxxx' （或创建.env文件）")
    print("\n前两项检查通过，环境搭建基本完成！配置密钥后重新运行本脚本完成全部验证。")
    sys.exit(0)

try:
    import llm_client
    answer = llm_client.chat(
        [{"role": "user", "content": "请只回复两个字：正常"}],
        temperature=0.1,
    )
    print(f"  ✓ API调用成功，模型回复: {answer.strip()}")
except Exception as e:
    print(f"  ✗ API调用失败: {e}")
    print("  请检查: 1)密钥是否正确 2)BASE_URL是否匹配 3)网络是否通畅")
    sys.exit(1)

# ---------------------------------------------------------------
# 检查4：结构化输出（chat_json）
# ---------------------------------------------------------------
print("-" * 50)
print("检查4：结构化输出 chat_json...")
try:
    from models import SearchPlan
    plan = llm_client.chat_json(
        [{"role": "user", "content": "我想调研'多模态大模型的幻觉缓解方法'这个主题"}],
        schema_class=SearchPlan,
        temperature=0.5,
    )
    print(f"  ✓ 结构化输出成功!")
    print(f"    生成的检索词: {plan.keywords}")
    print(f"    筛选标准: {plan.inclusion_criteria}")
    print(f"    目标篇数: {plan.target_count}")
except Exception as e:
    print(f"  ✗ 结构化输出失败: {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
llm_client.print_usage()
print("\n🎉 第0步环境验证全部通过！可以开始第1步（检索Agent）的开发了。")
