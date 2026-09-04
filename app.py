"""
app.py —— Streamlit前端（Demo展示界面）
================================
启动方法:
  conda activate brain
  cd d:\trae项目\literature_agent
  streamlit run app.py

界面结构:
  侧边栏: 主题输入 + 目标篇数 + 启动按钮 + 说明
  主区域:
    - 运行状态区: 四个Agent的实时进度
    - 结果标签页: 📄信息卡片 | ✅审查明细 | 📊综述表格 | 🕸引用图谱 | 📝最终报告
  复用按钮: 已有数据时可直接展示产出（不用重新跑流水线）
"""

import os
import sys
import json

import streamlit as st

# 项目根目录加入path（streamlit的工作目录可能不同）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import PaperMeta, PaperCard, ReviewResult

# 证据图片目录（方案一：证据链穿透）
EVIDENCE_DIR = os.path.join(config.PROJECT_ROOT, "evidence")

# ---------------------------------------------------------------
# 页面全局配置
# ---------------------------------------------------------------
st.set_page_config(
    page_title="科研文献整理Agent",
    page_icon="📚",
    layout="wide",
)

# 会话状态: 存放流水线的中间产出（跨标签页共享，避免重复计算）
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "run_outputs" not in st.session_state:
    st.session_state.run_outputs = {}


# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(path: str, cls=None):
    """读取data/或output/下的JSON（缓存，切标签页不重读）"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if cls:
        return [cls(**item) for item in data]
    return data


@st.cache_data(show_spinner=False)
def load_text(path: str):
    """读取文本文件（报告/表格/图谱HTML）"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------
# 流水线执行（封装四Agent，带进度显示）
# ---------------------------------------------------------------
def run_pipeline(topic: str, target_count: int):
    """执行 规划->检索->提取->审查->综合 完整流水线"""
    from agents import planner, searcher, extractor, reviewer, synthesizer
    import llm_client

    # ---- 1. 规划Agent ----
    status = st.status("🧠 规划Agent: 分解研究主题...", expanded=True)
    plan = planner.make_plan(topic)
    st.session_state.run_outputs["plan"] = plan
    status.update(label=f"🧠 规划Agent完成: {len(plan.keywords)}组检索词",
                  state="complete")

    # ---- 2. 检索Agent ----
    status = st.status("🔍 检索Agent: 检索文献中（arXiv限速,约需1-3分钟）...",
                       expanded=True)
    st.write("检索词: " + ", ".join(plan.keywords))
    papers = searcher.run(plan, results_per_query=8, top_k=target_count)
    if len(papers) == 0:
        status.update(label="❌ 检索失败: 没有获得可用论文", state="error")
        st.stop()
    st.write(f"获得 {len(papers)} 篇论文, PDF已下载")
    status.update(label=f"🔍 检索Agent完成: {len(papers)}篇论文", state="complete")

    # ---- 3. 提取Agent ----
    status = st.status("📄 提取Agent: 抽取信息卡片中（每篇约1分钟）...",
                       expanded=True)
    cards = extractor.run(papers)
    status.update(label=f"📄 提取Agent完成: {len(cards)}张卡片, "
                        f"{sum(len(c.claims) for c in cards)}条声明",
                  state="complete")

    # ---- 4. 审查Agent ----
    status = st.status("✅ 审查Agent: 两级核验中（程序对齐+语义审查）...",
                       expanded=True)
    results = reviewer.run(papers, cards)
    report = reviewer.generate_report(results)
    status.update(label=f"✅ 审查Agent完成: 幻觉率 {report.hallucination_rate:.1%}",
                  state="complete")

    # ---- 5. 综合Agent ----
    status = st.status("📝 综合Agent: 生成综述报告与引用图谱...", expanded=True)
    outputs = synthesizer.run(topic)
    status.update(label="📝 综合Agent完成: 报告+表格+图谱已生成", state="complete")

    # ---- 6. 证据定位（方案一：证据链穿透）----
    status = st.status("📎 证据定位: 定位引用到PDF页面并高亮...", expanded=True)
    from agents import evidence_locator
    cards_raw = json.load(open(
        os.path.join(config.DATA_DIR, "paper_cards.json"), encoding="utf-8"))
    reviews_raw = json.load(open(
        os.path.join(config.DATA_DIR, "review_results.json"),
        encoding="utf-8"))
    ev_index = evidence_locator.build_evidence_index(cards_raw, reviews_raw)
    located_n = sum(1 for v in ev_index.values() if v.get("located"))
    status.update(label=f"📎 证据定位完成: {located_n}条声明可穿透到PDF原文",
                  state="complete")

    # ---- 7. 矛盾检测（方案三：学术争议发现）----
    status = st.status("⚡ 矛盾检测: 在已核验声明间寻找学术争议...",
                       expanded=True)
    from agents import conflict_detector
    conflicts = conflict_detector.run()
    n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
    n_ten = len(conflicts) - n_con
    status.update(
        label=f"⚡ 矛盾检测完成: 直接矛盾{n_con}处 + 张力{n_ten}处",
        state="complete")

    # ★关键: 清除数据加载缓存, 否则标签页仍显示上一次主题的旧数据
    # （load_json/load_text按路径缓存, 流水线覆盖文件后缓存不会自动失效）
    load_json.clear()
    load_text.clear()

    st.session_state.pipeline_done = True
    st.session_state.run_outputs["topic"] = topic
    llm_client.print_usage()

    # ★关键: 强制页面重新执行。原因: 本脚本从上到下运行, 侧边栏按钮触发
    # 流水线时, 主区域的旧数据早已加载完毕; 不rerun的话本次显示仍是旧结果
    st.rerun()


# ---------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------
with st.sidebar:
    st.title("📚 科研文献整理Agent")
    st.caption("检索 → 提取 → 审查 → 综合\n带防幻觉核验的文献综述流水线")

    st.divider()
    topic = st.text_input(
        "研究主题",
        value="脉冲神经网络的高效训练方法",
        help="支持中文输入，系统会自动翻译成英文检索词",
    )
    target_count = st.slider("目标论文数", 3, 15, 5)

    col1, col2 = st.columns(2)
    with col1:
        # ★运行锁: 防止并发流水线（2026-09-04实测教训）
        # searcher.run()内不调用st.*函数，Streamlit无法中断旧运行——
        # 流水线执行中再点按钮会并发起新运行，多个运行同时轰击
        # OpenAlex触发429限流，速度反而暴慢。必须显式加锁。
        running = st.session_state.get("pipeline_running", False)
        if st.button("🚀 完整运行", type="primary", use_container_width=True,
                     disabled=running,
                     help="运行中请耐心等待，重复点击会触发API限流"):
            st.session_state.pipeline_running = True
            try:
                with st.spinner("流水线运行中..."):
                    run_pipeline(topic, target_count)
            finally:
                st.session_state.pipeline_running = False
        if running:
            st.info("⏳ 流水线正在执行中，请等待完成后再操作页面")
    with col2:
        if st.button("📂 载入已有结果", use_container_width=True,
                     help="直接展示data/和output/目录下的历史产出"):
            st.session_state.pipeline_done = True

    st.divider()
    st.subheader("项目说明")
    st.markdown(f"""
**四个Agent流水线**

1. 🧠 **规划** — 主题→检索计划
2. 🔍 **检索** — arXiv/S2双源+LLM粗筛
3. 📄 **提取** — PDF→带引用的信息卡片
4. ✅ **审查** — 两级核验防幻觉
5. 📝 **综合** — 只用已核验内容生成综述

**核心创新**: 每条声明强制附原文引用，
程序对齐+LLM语义双重核验，
幻觉率全程量化可审计。
**证据链穿透**: 通过核验的声明
可一键定位到PDF原文高亮位置。
**可信问答**: 只基于已核验声明回答，
证据不足明确拒答。
**矛盾检测**: 自动发现论文间的
观点冲突与学术张力。

*模型: {os.path.basename(config.MODEL_NAME)}*
""")

# ---------------------------------------------------------------
# 主区域: 结果展示标签页
# ---------------------------------------------------------------
# 显示本次结果对应的主题（从会话状态取; 载入历史结果时显示通用标题）
_result_topic = st.session_state.run_outputs.get("topic", "历史结果")
st.title(f"📚 研究综述: {_result_topic}")

if not st.session_state.pipeline_done:
    st.info("👈 在侧边栏输入研究主题，点击「完整运行」或「载入已有结果」开始")
    st.stop()

# 各标签页共用的数据加载
papers = load_json(os.path.join(config.DATA_DIR, "papers_meta.json"), PaperMeta)
cards = load_json(os.path.join(config.DATA_DIR, "paper_cards.json"), PaperCard)
reviews = load_json(os.path.join(config.DATA_DIR, "review_results.json"),
                    ReviewResult)
evidence_index = load_json(os.path.join(config.DATA_DIR,
                                        "evidence_index.json")) or {}
table_md = load_text(os.path.join(config.OUTPUT_DIR, "review_table.md"))
report_md = load_text(os.path.join(config.OUTPUT_DIR, "final_report.md"))
graph_html = load_text(os.path.join(config.OUTPUT_DIR, "citation_graph.html"))

if cards is None:
    st.warning("没有找到历史产出数据，请先完整运行一次流水线")
    st.stop()


# ---------------------------------------------------------------
# 证据链穿透面板（方案一核心UI组件）
# ---------------------------------------------------------------
def evidence_popover(label: str, arxiv_id: str, claim_index: int,
                     claim_content: str = "", quotes=None):
    """
    声明旁的"📎 证据"弹层按钮: 点击展开显示
    PDF高亮页面截图 + 引用原文 + 定位信息

    参数:
        label: 按钮文字
        arxiv_id/claim_index: 声明唯一标识(对应evidence_index的key)
        claim_content: 声明内容(面板顶部展示)
        quotes: 声明的引用列表(面板中部展示)
    """
    key = f"{arxiv_id}#{claim_index}"
    ev = evidence_index.get(key)

    with st.popover(label, use_container_width=False):
        st.markdown(f"**声明**: {claim_content or '(见上方)'}")

        if ev is None:
            st.info("该声明无证据索引（可能未通过核验，证据仅为核验通过的声明生成）")
            if quotes:
                st.markdown("**引用原文(未定位)**:")
                for q in quotes:
                    st.markdown(f"> `{q.section}` {q.text[:150]}")
            return

        if not ev.get("located"):
            st.warning("引用未能在PDF中定位（可能因文本截断或格式差异），"
                       "以下为提取时记录的引用原文")
            if quotes:
                for q in quotes:
                    st.markdown(f"> `{q.section}` {q.text[:150]}")
            return

        # 定位成功: 展示元信息+高亮截图
        page = ev["page"] + 1  # 显示为1-based页码
        st.markdown(
            f"出处: **{ev['pdf']}** 第{page}页 "
            f"| 匹配级别: {ev.get('match_level', '?')}"
        )
        img_path = os.path.join(EVIDENCE_DIR, ev["image"])
        if os.path.exists(img_path):
            st.image(img_path, caption=f"PDF第{page}页 · 黄色高亮=证据原文",
                     use_container_width=True)
        else:
            st.error(f"证据图片缺失: {ev['image']}")
        if quotes:
            st.markdown("**引用原文**:")
            for q in quotes:
                st.markdown(f"> `{q.section}` {q.text[:150]}")


# ---- 顶部统计卡片 ----
if reviews:
    total = len(reviews)
    good = sum(1 for r in reviews
               if r.quote_alignment and r.verdict.verdict == "supported")
    ev_located = sum(1 for v in evidence_index.values() if v.get("located"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("收录论文", f"{len(cards)} 篇")
    c2.metric("声明总数", f"{total} 条")
    c3.metric("通过核验", f"{good} 条")
    c4.metric("幻觉率", f"{(total - good) / total:.1%}")
    c5.metric("可穿透证据", f"{ev_located} 条")

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🔍 检索结果", "📄 信息卡片", "✅ 审查明细", "💬 可信问答", "⚡ 学术争议",
     "📊 综述表格", "🕸 引用图谱", "📝 最终报告"])

# ---- 标签0: 检索结果（论文排序输出面板） ----
with tab0:
    st.subheader("检索结果排序")
    st.caption(
        "排序依据四路信号加权: **相关性**(LLM判断主题切题度) + "
        "**RRF共识**(被多个检索角度共同命中) + **影响力**(被引数) + "
        "**引用链**(被种子论文参考文献共同引用的次数，奠基经典专属信号)。\n"
        "经典论文保障机制: 共被引≥3走**免检通道**(学术界共识优先于LLM打分)；"
        "共被引≥2享受粗筛阈值补偿；最老的经典(如SpikeProp)由规划Agent的"
        "**LLM领域先验**按标题注入，弥补引用图谱数据缺失。"
    )

    # 排序方式切换（同一份论文池，不同视角的输出方式）
    sort_mode = st.radio(
        "排序方式",
        ["🎯 综合推荐", "🤖 相关优先", "🔥 经典优先", "🆕 最新优先"],
        horizontal=True,
        help="综合推荐=四路信号加权(默认); 相关优先=LLM切题度优先; "
             "经典优先=被引数优先; 最新优先=发表年份优先",
    )
    if papers:
        if sort_mode == "🎯 综合推荐":
            shown = sorted(papers, key=lambda p: p.final_score or 0,
                           reverse=True)
        elif sort_mode == "🤖 相关优先":
            shown = sorted(papers, key=lambda p: p.relevance_score or 0,
                           reverse=True)
        elif sort_mode == "🔥 经典优先":
            shown = sorted(papers, key=lambda p: p.cited_by, reverse=True)
        else:
            shown = sorted(papers, key=lambda p: p.year, reverse=True)

        for rank, p in enumerate(shown, 1):
            # 徽章: 让用户一眼看懂"为什么这篇排在这里"
            badges = []
            if p.chain_hits >= 3:
                badges.append(f"🔗引用链经典(被{p.chain_hits}篇种子引用)")
            elif p.chain_hits > 0:
                badges.append(f"🔗引用链({p.chain_hits})")
            if p.cited_by >= 300:
                badges.append(f"🔥高被引({p.cited_by})")
            if (p.relevance_score or 0) >= 4.5:
                badges.append(f"🤖高度切题({p.relevance_score})")
            badge_str = " ".join(f"`{b}`" for b in badges)

            score_parts = (
                f"相关性 {p.relevance_score if p.relevance_score is not None else '免检'}"
                f" · 被引 {p.cited_by} · 引用链 {p.chain_hits}"
                + (f" · 综合分 {p.final_score}" if p.final_score else "")
            )
            with st.expander(
                f"{rank}. {p.title[:70]} ({p.year}) {badge_str}"
            ):
                st.caption(f"[{p.arxiv_id}] {score_parts}")
                st.markdown(
                    p.abstract[:400] + ("..." if len(p.abstract) > 400 else "")
                )
    else:
        st.info("暂无检索结果数据")

# ---- 标签1: 信息卡片 ----
with tab1:
    st.subheader("论文信息卡片（提取Agent产出）")
    st.caption("点击每条声明旁的 📎 按钮可穿透查看PDF原文高亮位置")
    for card in cards:
        with st.expander(f"📁 {card.title[:55]}... [{card.arxiv_id}] "
                         f"({card.year}) — {card.method_category}"):
        # 展开显示每条声明及其原文引用
            for i, claim in enumerate(card.claims):
                color = {"method": "🔵", "result": "🟢",
                         "limitation": "🟠"}.get(claim.claim_type, "⚪")
                # 声明 + 证据弹层按钮同行布局
                c_col, e_col = st.columns([5, 1])
                with c_col:
                    st.markdown(f"{color} **[{claim.claim_type}]** "
                                f"{claim.content}")
                with e_col:
                    evidence_popover("📎 证据", card.arxiv_id, i,
                                     claim.content, claim.quotes)
                for q in claim.quotes:
                    st.markdown(
                        f"> `{q.section}` {q.text[:200]}..."
                        if len(q.text) > 200 else f"> `{q.section}` {q.text}"
                    )
                st.divider()

# ---- 标签2: 审查明细 ----
with tab2:
    st.subheader("两级核验明细（审查Agent产出）")
    st.caption("第1级: 程序化引用对齐(抓伪造引用) | "
               "第2级: LLM语义核验(抓夸大/曲解)")

    # 用表格展示全部核验结果
    rows = []
    for r in reviews:
        rows.append({
            "论文": r.arxiv_id,
            "#": r.claim_index,
            "声明": r.claim_content[:50] + "...",
            "引用对齐": "✓" if r.quote_alignment else "✗",
            "语义判定": r.verdict.verdict,
            "置信度": f"{r.verdict.confidence:.2f}",
            "理由": r.verdict.reason[:60],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # 通过核验的声明: 带证据穿透按钮
    good_list = [r for r in reviews
                 if r.quote_alignment and r.verdict.verdict == "supported"]
    st.markdown(f"### ✓ 通过核验的声明 ({len(good_list)}条)")
    st.caption("点击 📎 可穿透查看该声明在PDF原文中的高亮位置")
    # 声明内容需要从cards里取（reviews只有claim_content摘要）
    for r in good_list:
        c_col, e_col = st.columns([5, 1])
        with c_col:
            st.markdown(f"✅ **[{r.arxiv_id} #{r.claim_index}]** "
                        f"{r.claim_content[:70]}...")
        with e_col:
            # 找对应claim的quotes（从cards索引）
            quotes = None
            for card in cards:
                if card.arxiv_id == r.arxiv_id and r.claim_index < len(card.claims):
                    quotes = card.claims[r.claim_index].quotes
                    break
            evidence_popover("📎 证据", r.arxiv_id, r.claim_index,
                             r.claim_content, quotes)

    # 分组展示: 通过 vs 剔除
    bad = [r for r in reviews if not r.quote_alignment
           or r.verdict.verdict != "supported"]
    st.markdown(f"### 被剔除的声明 ({len(bad)}条)")
    for r in bad:
        with st.expander(f"❌ [{r.arxiv_id} #{r.claim_index}] "
                         f"{r.claim_content[:50]}..."):
            st.markdown(f"- 引用对齐: {'✓ 真实' if r.quote_alignment else '✗ 疑似伪造'}")
            st.markdown(f"- 语义判定: **{r.verdict.verdict}** "
                        f"(置信度 {r.verdict.confidence})")
            st.markdown(f"- 判定理由: {r.verdict.reason}")

# ---- 标签3: 可信问答（方案二）----
with tab3:
    st.subheader("💬 可信问答（Evidence-Bounded QA）")
    st.caption("只基于已通过两级核验的声明池回答，证据不足时明确拒答——"
               "宁可拒答，不可编造")

    from agents import qa_agent as qa_mod

    # 会话状态: 对话历史
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # 左右布局: 左侧对话区 + 右侧声明池浏览
    qa_col, pool_col = st.columns([3, 2])

    with qa_col:
        # 渲染历史对话
        for msg in st.session_state.qa_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # 回答消息附引用声明与拒答说明
                if msg["role"] == "assistant" and msg.get("used_claims"):
                    st.caption(f"依据: 声明 {msg['used_claims']}")
                if msg["role"] == "assistant" and msg.get("missing"):
                    st.warning(f"缺失证据: {msg['missing']}")

        # 输入框
        if q := st.chat_input("问点什么…（如: HorNet在ImageNet上的表现如何？）"):
            st.session_state.qa_history.append(
                {"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("检索声明池并生成有据回答..."):
                    try:
                        ans, _ = qa_mod.answer_question(
                            q, st.session_state.qa_history[:-1])
                        body = ans.answer
                        if not ans.can_answer:
                            body = (f"⚠️ {ans.answer}\n\n"
                                    f"**缺失的证据**: {ans.missing}")
                        st.markdown(body)
                        if ans.used_claims:
                            st.caption(f"依据: 声明 {ans.used_claims}")
                        st.session_state.qa_history.append({
                            "role": "assistant",
                            "content": body,
                            "used_claims": ans.used_claims,
                            "missing": ans.missing if not ans.can_answer else "",
                        })
                    except Exception as e:
                        st.error(f"问答失败: {e}")

    with pool_col:
        st.markdown("**📚 当前声明池**（问答的知识边界）")
        pool = qa_mod.build_claim_pool()
        if pool:
            st.caption(f"共{len(pool)}条已核验声明 · 回答只能引用这些内容")
            for c in pool:
                type_icon = {"method": "🔵", "result": "🟢",
                             "limitation": "🟠"}.get(c["type"], "⚪")
                st.markdown(
                    f"{type_icon} **[#{c['num']}]** {c['content'][:80]}..."
                    if len(c["content"]) > 80
                    else f"{type_icon} **[#{c['num']}]** {c['content']}"
                )
        else:
            st.info("声明池为空，请先运行完整流水线")

        # 拒答能力一键自检（E5实验入口）
        st.divider()
        if st.button("🧪 拒答能力自检", help="运行3个可答+3个知识库外问题，"
                     "验证系统的拒答可靠性"):
            with st.spinner("测试中（约1分钟）..."):
                r = qa_mod.run_refusal_test()
            if r:
                m1, m2 = st.columns(2)
                m1.metric("可答组正确率", f"{r['answerable_acc']:.0%}")
                m2.metric("拒答组正确率", f"{r['refusal_acc']:.0%}")
                with st.expander("测试明细"):
                    for d in r["details"]:
                        mark = "✓" if d["correct"] else "✗"
                        st.markdown(
                            f"{mark} {'应答' if d['expect_answerable'] else '应拒'}"
                            f"→{'答了' if d['actual_answerable'] else '拒答'} "
                            f"| {d['question']}")

# ---- 标签4: 学术争议（方案三）----
with tab4:
    st.subheader("⚡ 学术争议（矛盾检测Agent产出）")
    st.caption("在已核验声明间自动检测观点冲突——"
               "🔴 直接矛盾(结论对立) | 🟡 张力(条件不同导致的表象冲突)。"
               "每条争议声明可点击📎穿透到PDF原文")

    conflicts = load_json(os.path.join(config.DATA_DIR, "conflicts.json"))

    if not conflicts:
        st.info("当前声明池未检测到学术争议（论文主题较分散时属正常）。"
                "论文数更多、主题更聚焦时争议检出率更高。")
        # 手动触发按钮（历史数据没有conflicts.json时用）
        if st.button("⚡ 立即检测矛盾"):
            with st.spinner("检测中（预筛+LLM精判，约1-3分钟）..."):
                from agents import conflict_detector
                conflict_detector.run()
                load_json.clear()
                st.rerun()
    else:
        n_con = sum(1 for c in conflicts if c["relation"] == "contradict")
        n_ten = len(conflicts) - n_con
        m1, m2 = st.columns(2)
        m1.metric("直接矛盾", f"{n_con} 处")
        m2.metric("张力关系", f"{n_ten} 处")

        for c in conflicts:
            icon = "🔴" if c["relation"] == "contradict" else "🟡"
            sev = "★" * c["severity"] + "☆" * (5 - c["severity"])
            with st.expander(
                f"{icon} [{c['relation']}] {c['topic']} "
                f"(严重度 {sev}) — {c['explanation'][:40]}..."
            ):
                st.markdown(f"**分析**: {c['explanation']}")

                # 双方观点对照展示（各带证据穿透）
                for side, label in [("a", "观点A"), ("b", "观点B")]:
                    s = c[side]
                    st.markdown(f"**{label}**: "
                                f"《{s['paper_title'][:40]}》"
                                f"({s['year']}, arXiv:{s['arxiv_id']})")
                    s_col, e_col = st.columns([5, 1])
                    with s_col:
                        st.markdown(f"> {s['content']}")
                    with e_col:
                        evidence_popover(
                            "📎 证据", s["arxiv_id"], s["claim_index"],
                            s["content"])
                st.divider()

        # 重新检测按钮（数据更新后刷新）
        if st.button("🔄 重新检测"):
            with st.spinner("检测中..."):
                from agents import conflict_detector
                conflict_detector.run()
                load_json.clear()
                st.rerun()

# ---- 标签5: 综述表格 ----
with tab5:
    st.subheader("方法分类对比表（仅收录通过核验的声明）")
    if table_md:
        st.markdown(table_md)
    else:
        st.warning("综述表格未生成")

# ---- 标签6: 引用图谱 ----
with tab6:
    st.subheader("引用关系图谱（交互式）")
    st.caption("🟢 本次检索论文 | 🔴 内部互引箭头 | ⚪ 外部被引论文 | "
               "橙色=方法分类(降级模式)")
    if graph_html:
        # 用iframe嵌入pyvis生成的HTML（组件化渲染交互图）
        st.components.v1.html(graph_html, height=620)
    else:
        st.warning("引用图谱未生成")

# ---- 标签7: 最终报告 ----
with tab7:
    st.subheader("最终综述报告")
    if report_md:
        st.markdown(report_md)
        # 提供下载按钮
        st.download_button(
            "⬇️ 下载完整报告 (Markdown)",
            data=report_md,
            file_name="literature_review.md",
            mime="text/markdown",
        )
    else:
        st.warning("最终报告未生成")
