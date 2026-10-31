# Multi-Agent Debate 프레임워크
# Stage 1 (HoQ Construction) + Stage 2 (Conflict Resolution) 그래프 통합 관리
#
# Stage 1 구조:
#   Phase 2-A: EC 리스트 토론 (engineering_expert ↔ market_expert, 도구 호출 포함)
#   Phase 2-B: Roof Matrix 토론 (engineering_expert ↔ market_expert)
#   Phase 3:   세그먼트별 Matrix Fill (공유 EC × 각 세그먼트 CR)
import re
import json, os, time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

import arxiv
from tavily import TavilyClient
from serpapi import GoogleSearch

import settings
from src.schemas.state import QFDState
from src.prompts import (
    ENGINEERING_EXPERT_SYSTEM,
    MARKET_EXPERT_SYSTEM,
    ENGINEERING_EXPERT_ROOF_SYSTEM,
    MARKET_EXPERT_ROOF_SYSTEM,
    build_matrix_fill_reasoning_system,
    build_matrix_fill_json_system,
    SEGMENT_ADVOCATE_REASONING_SYSTEM,
    SEGMENT_ADVOCATE_JSON_SYSTEM,
    ORCHESTRATOR_SYSTEM,
    MARKET_EXPERT_JUDGE_SYSTEM,
    ENGINEERING_EXPERT_JUDGE_SYSTEM,
    JUDGE_ADJUSTMENT_TRANSLATION_SYSTEM,
    ENGINEERING_EXPERT_PC_SYSTEM,
    MARKET_EXPERT_PC_SYSTEM,
    PC_SPECIALIZATION_BLOCK,
    TARGET_VALUE_MODEL_CONTEXT,
    build_pc_matrix_fill_reasoning_system,
    build_pc_matrix_fill_json_system,
    ENGINEERING_EXPERT_PC_ROOF_SYSTEM,
    MARKET_EXPERT_PC_ROOF_SYSTEM,
    TARGET_VALUE_ENGINEERING_SYSTEM,
    TARGET_VALUE_MARKET_SYSTEM,
    PC_SPEC_SYSTEM,
)

# ═════════════════════════════════════════════════════════════════════════════
# LLM & Embeddings
# ═════════════════════════════════════════════════════════════════════════════

LOG_DIR = f"{settings.INTERMEDIATE_PATH}/logs"


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _format_message_for_log(msg: dict) -> str:
    """단일 메시지를 읽기 좋은 텍스트로 변환"""
    agent = msg.get("agent", "unknown")
    round_num = msg.get("round", "?")
    phase = msg.get("phase", "")
    phase_str = f" [{phase}]" if phase else ""

    header = f"{'='*70}\n[Round {round_num}] {agent}{phase_str}\n{'='*70}"
    content = msg.get("content", {})
    if isinstance(content, dict):
        body = json.dumps(content, indent=2, ensure_ascii=False)
    else:
        body = str(content)
    return f"{header}\n{body}\n"


def save_debate_log(filename: str, messages: list[dict], header: str = "") -> str:
    """토론 메시지를 readable .txt 파일로 저장

    Returns:
        저장된 파일 경로
    """
    _ensure_log_dir()
    path = f"{LOG_DIR}/{filename}"

    lines = []
    if header:
        lines.append(header)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

    for msg in messages:
        lines.append(_format_message_for_log(msg))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [Log] 토론 로그 저장: {path}")
    return path


_client = OpenAI()
_embeddings = OpenAIEmbeddings()
_chat_llm = ChatOpenAI(model=settings.MODEL, service_tier=settings.SERVICE_TIER)


def _get_review_db(segment: str) -> Chroma:
    return Chroma(
        collection_name=f"review_{segment.lower()}",
        embedding_function=_embeddings,
        persist_directory=f"{settings.CHROMA_BASE_PATH}/chroma_db_review_{segment.lower()}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tool Definitions — @tool 데코레이터로 에이전트가 직접 호출
# ═════════════════════════════════════════════════════════════════════════════

@tool
def search_arxiv(query: str) -> str:
    """Search ArXiv for technical papers on laptop engineering topics.
    Use this to find academic research on thermal management, battery technology,
    display engineering, CPU/GPU performance, materials science, etc.
    Returns paper titles and summary excerpts."""
    client = arxiv.Client(
        page_size=3,
        delay_seconds=3.0,   # ArXiv 공식 권장: 3초 간격
        num_retries=3,
    )
    search = arxiv.Search(
        query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance
    )
    for attempt in range(3):
        try:
            lines = []
            for paper in client.results(search):
                lines.append(f"[{paper.title}] {paper.summary[:300]}")
            return "\n\n".join(lines) if lines else "No results found."
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 5 * (attempt + 1)   # 5초, 10초 대기 후 재시도
                time.sleep(wait)
                continue
            return f"ArXiv search error: {e}"


@tool
def search_google(query: str) -> str:
    """Search Google via SerpAPI for technical specifications, product benchmarks,
    patents, datasheets, and engineering references.
    Returns titles and snippets from top results."""
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return "SerpAPI key not configured."
    try:
        search = GoogleSearch({"q": query, "api_key": api_key, "num": 3})
        results = search.get_dict().get("organic_results", [])
        lines = []
        for r in results[:3]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            lines.append(f"[{title}] {snippet[:300]}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"SerpAPI search error: {e}"


@tool
def search_tavily(query: str) -> str:
    """Search Tavily for market trends, consumer preferences, product reviews,
    and industry news about laptops.
    Returns titles and content excerpts from top results."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "Tavily API key not configured."
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query, search_depth="basic", max_results=3
        )
        lines = []
        for r in response.get("results", [])[:3]:
            title = r.get("title", "")
            content = r.get("content", "")
            lines.append(f"[{title}] {content[:300]}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Tavily search error: {e}"


@tool
def search_reviews(segment_and_query: str) -> str:
    """Search the ChromaDB review database for raw user reviews relevant to a topic.
    Input format: 'segment|query' where segment is 'Professional' or 'Casual'.
    Example: 'Professional|battery life during heavy workload'
    Returns actual user review excerpts from Reddit."""
    try:
        parts = segment_and_query.split("|", 1)
        if len(parts) != 2:
            return "Error: format must be 'segment|query'. Example: 'Professional|battery life'"
        segment, query = parts[0].strip(), parts[1].strip()
        db = _get_review_db(segment)
        results = db.similarity_search(query, k=5)
        lines = []
        for doc in results:
            text = doc.page_content.split(" → ")[0].strip()
            lines.append(f'"{text[:300]}"')
        return "\n\n".join(lines) if lines else "No matching reviews found."
    except Exception as e:
        return f"Review search error: {e}"


# 에이전트별 도구 세트
ENGINEERING_TOOLS = [search_arxiv, search_google, search_reviews]
MARKET_TOOLS = [search_tavily, search_google, search_reviews]


# ═════════════════════════════════════════════════════════════════════════════
# Shared Helpers (Matrix Fill용 — 도구 없이 직접 호출)
# ═════════════════════════════════════════════════════════════════════════════

def _llm_json(system: str, user: str) -> dict:
    """도구 없이 JSON 응답만 받는 직접 호출 (Matrix Fill용)"""
    resp = _client.chat.completions.create(
        model=settings.MODEL,
        service_tier=settings.SERVICE_TIER,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _llm_text(system: str, user: str) -> str:
    """도구 없이 자유 텍스트 응답을 받는 직접 호출 (CoT Reasoning용)"""
    resp = _client.chat.completions.create(
        model=settings.MODEL,
        service_tier=settings.SERVICE_TIER,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def _format_crs(cr_list: list[dict]) -> str:
    lines = []
    for cr in cr_list:
        lines.append(f'  - "{cr["cr"]}" (priority: {cr["priority"]})')
    return "\n".join(lines)


def _retrieve_reviews(segment: str, cr_list: list[dict], k_per_cr: int = 3) -> str:
    """CR별로 ChromaDB에서 관련 원문 리뷰를 검색하여 포맷 (Matrix Fill용)"""
    db = _get_review_db(segment)
    seen = set()
    lines = []
    for cr in cr_list:
        label = cr["cr"]
        try:
            results = db.similarity_search(label, k=k_per_cr)
            for doc in results:
                text = doc.page_content.split(" → ")[0].strip()
                if text not in seen:
                    seen.add(text)
                    lines.append(f'  - "{text[:300]}"')
        except Exception:
            continue
    return "\n".join(lines) if lines else ""


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: EC Debate (Phase 2-A) — EC 리스트 토론
# 에이전트가 도구를 직접 호출하여 근거 수집 후 EC 제안/검토
# ═════════════════════════════════════════════════════════════════════════════

VALID_REL_VALUES = {-9, -3, 0, 3, 9}
VALID_IMPORTANCE_VALUES = {0, 3, 9}   # Judge 조정값 척도 (문서 2-1-1·2-3-3)
VALID_ROOF_VALUES = {-0.9, -0.5, 0.5, 0.9}

MIN_EC_DEBATE_ROUNDS = 2   # EC 토론 최소 라운드
MIN_ROOF_DEBATE_ROUNDS = 2  # Roof 토론 최소 라운드


def _snap_to_valid(value: int | float, valid_set: set) -> int | float:
    """가장 가까운 허용값으로 보정"""
    return min(valid_set, key=lambda x: abs(x - value))


def _validate_roof(result: dict) -> dict:
    """Roof matrix 값을 허용 범위로 보정 (Stage 1 Roof · Stage 3 PC Roof 공용)

    발송 9 #11: 0 / null / 비수치는 "상관 없음 = 미기재"이므로 항목을 삭제한다.
    (종전에는 null→0.0 저장, 0→−0.5 보정으로 없는 약상관이 생겼음)
    허용값 밖의 수치는 최근접 허용값으로 보정한다."""
    snapped = 0
    dropped = 0
    roof = result.get("roof_matrix", {})
    for ec_a in list(roof.keys()):
        row = roof[ec_a]
        if not isinstance(row, dict):
            del roof[ec_a]
            dropped += 1
            continue
        for ec_b in list(row.keys()):
            v = row[ec_b]
            if v is None or not isinstance(v, (int, float)) or v == 0:
                del row[ec_b]
                dropped += 1
            elif v not in VALID_ROOF_VALUES:
                row[ec_b] = _snap_to_valid(v, VALID_ROOF_VALUES)
                snapped += 1
        if not row:
            del roof[ec_a]
    if snapped or dropped:
        print(f"  ⚠️  Roof: 값 보정 {snapped}개, 0/null 항목 삭제 {dropped}개")
    return result


def _validate_relationship(result: dict) -> dict:
    """Relationship matrix 값을 허용 범위로 보정"""
    violations = 0
    rel = result.get("relationship_matrix", {})
    for cr in rel:
        for ec in rel[cr]:
            v = rel[cr][ec]
            if v is None:
                rel[cr][ec] = 0
                violations += 1
            elif v not in VALID_REL_VALUES:
                rel[cr][ec] = _snap_to_valid(v, VALID_REL_VALUES)
                violations += 1
    if violations > 0:
        print(f"  ⚠️  Relationship: {violations}개 값 보정됨")
    return result


def _parse_json_from_ai(message: AIMessage) -> dict:
    """AIMessage content에서 JSON을 파싱. 코드블록 래핑도 처리."""
    text = message.content.strip()
    # ```json ... ``` 래핑 제거
    if text.startswith("```"):
        lines = text.split("\n")
        # 첫 줄(```json)과 마지막 줄(```) 제거
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


# ─── EC Debate 노드 ─────────────────────────────────────────────────────────

def _compute_ec_range(state: QFDState) -> tuple[int, int]:
    """CR 수에 연동하여 EC 수 범위를 결정 (상한은 settings.MAX_ECS)"""
    # Common CR은 두 리스트에 모두 들어 있으므로 중복 없이 셈 (발송 10 A4)
    total_crs = len(_dedupe_crs(state["cr_list_a"], state["cr_list_b"]))
    max_ecs = min(settings.MAX_ECS, max(12, total_crs))
    min_ecs = max(8, max_ecs // 2)
    return min_ecs, max_ecs


def _check_compound_ecs(ec_list: list[dict]) -> list[str]:
    """복합 EC 탐지 — 경고 목록 반환"""
    warnings = []
    compound_signals = [" and ", " & ", "; "]
    for ec in ec_list:
        name = ec.get("ec", "")
        for signal in compound_signals:
            if signal.lower() in name.lower():
                warnings.append(
                    f"복합 EC 의심: '{name}' (signal: '{signal.strip()}')"
                )
                break
    return warnings


# ── 발송 9 #22: 빈 행(coverage) 검사 — Stage 1 EC 토론 · Stage 3 PC 토론 공용 ──

def _dedupe_crs(*lists: list[dict]) -> list[dict]:
    """여러 CR 리스트를 텍스트 기준으로 중복 없이 합침 (Common CR은 A·B 양쪽에 있음)"""
    seen: set[str] = set()
    out: list[dict] = []
    for lst in lists:
        for cr in lst:
            if cr["cr"] not in seen:
                seen.add(cr["cr"])
                out.append(cr)
    return out


def _format_crs_with_ids(cr_list: list[dict], name_to_id: dict[str, str]) -> str:
    return "\n".join(
        f'  {name_to_id.get(cr["cr"], "?")}: "{cr["cr"]}" (priority: {cr["priority"]})'
        for cr in cr_list
    )


def _resolve_quiet(key: str, id_map: dict[str, str]) -> str | None:
    """ID('EC-3') 또는 정본 이름을 정본 이름으로. 실패 시 None (경고 없음)"""
    k = str(key).strip()
    if k in id_map:
        return id_map[k]
    canonical = set(id_map.values())
    if k in canonical:
        return k
    ku = k.upper().replace(" ", "")
    for cid, name in id_map.items():
        if ku == cid.upper().replace(" ", ""):
            return name
    return None


def _coverage_gaps(
    items: list[dict], covers_key: str, target_ids: dict[str, str],
) -> tuple[list[str], int]:
    """각 item의 covers(ID 목록)를 정본 이름으로 치환(제자리)하고 미커버 대상을 반환.

    Returns:
        (미커버 대상 정본 이름 목록 — target_ids 순서, covers 필드가 없던 item 수)
    """
    covered: set[str] = set()
    missing_field = 0
    for it in items:
        cov = it.get(covers_key)
        if not isinstance(cov, list):
            missing_field += 1
            it[covers_key] = []
            continue
        resolved: list[str] = []
        for c in cov:
            name = _resolve_quiet(c, target_ids)
            if name and name not in resolved:
                resolved.append(name)
                covered.add(name)
        it[covers_key] = resolved
    uncovered = [name for name in target_ids.values() if name not in covered]
    return uncovered, missing_field


def _format_gap_block(
    gaps: list[str], name_to_id: dict[str, str], target_label: str, item_label: str,
    extra: dict[str, str] | None = None,
) -> str:
    """다음 라운드 프롬프트에 넣는 미커버 목록 블록"""
    if not gaps:
        return ""
    lines = [
        f"### ⚠️ {target_label}s NOT covered by any {item_label} in your last proposal "
        f"({len(gaps)}) — every {target_label} must map to at least one {item_label}:"
    ]
    for g in gaps:
        tail = f" ({extra[g]})" if extra and g in extra else ""
        lines.append(f'  - {name_to_id.get(g, "?")}: "{g}"{tail}')
    lines.append(
        f"Add or revise {item_label}s so these are covered, or justify each exception "
        f"in \"uncovered_{target_label.lower()}s\"."
    )
    return "\n".join(lines) + "\n"


def engineering_expert_ec_node(state: QFDState) -> dict:
    """Engineering Expert: EC 리스트 제안/수정 (도구 호출 가능)

    발송 9 #22: EC마다 담당 CR ID(covers)를 제출받아 빈 행(미커버 CR)을 라운드 안에서
    검사한다. 미커버 CR이 있으면 복합 EC 검사와 같은 방식으로 finalized를 거부하고
    다음 라운드에 목록을 주입한다. 마지막 라운드에는 채택하되 기록한다.
    """
    seg_a = state["segment_a"]
    seg_b = state["segment_b"]
    cr_all = _dedupe_crs(state["cr_list_a"], state["cr_list_b"])
    cr_ids = _make_id_map([c["cr"] for c in cr_all], "CR")
    cr_name_to_id = {v: k for k, v in cr_ids.items()}
    cr_text_a = _format_crs_with_ids(state["cr_list_a"], cr_name_to_id)
    cr_text_b = _format_crs_with_ids(state["cr_list_b"], cr_name_to_id)
    min_ecs, max_ecs = _compute_ec_range(state)
    prev_gaps = state.get("coverage_gaps", []) or []
    gap_block = _format_gap_block(prev_gaps, cr_name_to_id, "CR", "EC")

    if state["iteration"] == 0:
        user_content = f"""Propose a unified set of Engineering Characteristics (ECs) for a laptop product.
This EC set is shared across ALL user segments — every segment uses the SAME ECs.

### {seg_a} Customer Requirements ({len(state['cr_list_a'])}):
{cr_text_a}

### {seg_b} Customer Requirements ({len(state['cr_list_b'])}):
{cr_text_b}

### Task:
FIRST: Use your search tools to gather technical evidence relevant to the CRs above.
Search for specific topics (e.g., "laptop battery Wh capacity benchmark", "laptop thermal
throttling sustained performance") — NOT generic queries.

THEN: Based on the evidence, propose ECs that cover both segments' CRs.
The acceptable range is {min_ecs}–{max_ecs} ECs.
Include an EC ONLY if it is genuinely needed to capture a distinct, measurable engineering trade-off.
Each EC must be a SINGLE measurable engineering parameter with ONE unit.

### Coverage — no empty rows:
For each EC, list in "covers" the CR IDs (e.g. "CR-3") it primarily serves.
Every CR ID above must appear in at least one EC's "covers". If the EC limit
makes that impossible, list the exceptions in "uncovered_crs" with a reason.
The {min_ecs}–{max_ecs} limit takes PRECEDENCE over coverage: when it binds,
cover higher-Priority CRs first and leave lower-Priority CRs uncovered.

Output JSON:
{{
    "ec_list": [{{"ec": "single measurable engineering parameter (unit)", "covers": ["CR-3", "CR-7"]}}],
    "uncovered_crs": [{{"cr": "CR-n", "reason": "..."}}],
    "reasoning": "Explanation of how the EC set covers both segments, citing tool results"
}}"""
    else:
        feedback = [m for m in state["messages"] if m["agent"] == "market_expert"][-1]
        cur_ecs = state.get("ec_list", [])
        cur_ec_text = "\n".join(
            f'  EC-{i}: "{ec["ec"]}"  ← covers: '
            + (", ".join(cr_name_to_id.get(c, "?") for c in ec.get("covers", [])) or "(none)")
            for i, ec in enumerate(cur_ecs, 1)
        )
        user_content = f"""The Market Expert reviewed your EC proposal:

{json.dumps(feedback["content"], indent=2, ensure_ascii=False)}

### Your current EC list ({len(cur_ecs)}):
{cur_ec_text}

{gap_block}
Revise THIS list according to the feedback — do not rebuild it from scratch.
Keep every EC you do not explicitly change, and output the full revised list.

### CRITICAL EVALUATION RULES:
For EACH item in "ec_feedback", you MUST:
1. Use search tools to find COUNTER-EVIDENCE against the suggestion.
2. Use search tools to find SUPPORTING evidence for your current design.
3. Compare both sides, then decide:
   - ACCEPT: "[feedback point]" → Change: [specific EC modification]
   - REJECT: "[feedback point]" → Reason: [engineering evidence from tool results]

Accepting ALL feedback without REJECT is a sign of insufficient engineering judgment.

### {seg_a} Customer Requirements ({len(state['cr_list_a'])}):
{cr_text_a}

### {seg_b} Customer Requirements ({len(state['cr_list_b'])}):
{cr_text_b}

Stay within {min_ecs}–{max_ecs} ECs total.
If two ECs measure essentially the same trade-off, merge them.
Ensure every EC is a SINGLE measurable engineering parameter — no compound ECs.
Coverage: for each EC list in "covers" the CR IDs it primarily serves. Every CR ID
should appear in at least one EC's "covers" — a CR with no EC is an empty HoQ row.
The {min_ecs}–{max_ecs} limit takes PRECEDENCE over coverage: when it binds, cover
higher-Priority CRs first and list the remaining CR IDs in "uncovered_crs" with a reason.
Set "finalized": true ONLY when all feedback has been addressed, the EC count is within
{min_ecs}–{max_ecs}, AND every CR is covered or justified in "uncovered_crs".

Output JSON:
{{
    "ec_list": [{{"ec": "single measurable engineering parameter (unit)", "covers": ["CR-3", "CR-7"]}}],
    "uncovered_crs": [{{"cr": "CR-n", "reason": "..."}}],
    "feedback_responses": [{{"feedback": "original feedback", "action": "ACCEPT|REJECT", "detail": "what changed or why rejected"}}],
    "reasoning": "Summary of changes and evidence",
    "finalized": true | false
}}"""

    llm_with_tools = _chat_llm.bind_tools(ENGINEERING_TOOLS)
    messages = [
        {"role": "system", "content": ENGINEERING_EXPERT_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    # 도구 호출 루프: 에이전트가 도구를 호출하면 결과를 돌려주고 다시 호출
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 6  # 무한 루프 방지

    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [eng] {tool_name}({list(tool_args.values())[0][:60]}...)")

            # 도구 실행
            tool_map = {t.name: t for t in ENGINEERING_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [eng] 도구 {tool_call_count}회 호출 완료")

    # Round 2+에서 도구 미호출 경고
    if state["iteration"] > 0 and tool_call_count == 0:
        print("    ⚠️  Round 2+에서 도구 미호출 — 근거 없이 피드백 수용 가능성")

    # JSON 파싱
    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 재시도 (json_object 모드로)
        result = _llm_json(
            ENGINEERING_EXPERT_SYSTEM,
            user_content + "\n\nIMPORTANT: Output ONLY valid JSON. No other text.",
        )

    # EC 수 범위 검증
    ec_count = len(result.get("ec_list", []))
    if ec_count < min_ecs or ec_count > max_ecs:
        print(f"    ⚠️  EC 수 {ec_count}개 (허용 범위: {min_ecs}–{max_ecs})")

    # 복합 EC 검증
    compound_warnings = _check_compound_ecs(result.get("ec_list", []))
    for w in compound_warnings:
        print(f"    ⚠️  {w}")

    # serves 필드가 있으면 제거 (하위 호환) + 이름 정규화: LLM이 표시용 "EC-n: " 접두를
    # 이름에 복사하는 경우가 있어 제거 (10/1 R3 실측)
    for ec in result.get("ec_list", []):
        ec.pop("serves", None)
        if isinstance(ec.get("ec"), str):
            ec["ec"] = re.sub(r"^\s*EC-\d+\s*:\s*", "", ec["ec"]).strip()

    # 빈 행 검사 (발송 9 #22): covers ID → 정본 CR 이름 치환, 미커버 CR 산출
    gaps, n_nocov = _coverage_gaps(result.get("ec_list", []), "covers", cr_ids)
    if n_nocov:
        print(f"    ⚠️  covers 필드 없는 EC {n_nocov}개")
    if gaps:
        print(f"    ⚠️  미커버 CR {len(gaps)}개 (빈 행): "
              + ", ".join(cr_name_to_id.get(g, "?") for g in gaps))
    else:
        print(f"    ✅ 커버리지: CR {len(cr_ids)}개 전부 EC 배정")

    update = {
        "ec_list": result.get("ec_list", []),
        "coverage_gaps": gaps,
        "iteration": state["iteration"] + 1,
        "messages": [
            {
                "agent": "engineering_expert",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }

    # EC 수 범위 위반 여부
    ec_out_of_range = ec_count < min_ecs or ec_count > max_ecs

    # 최소 라운드 강제: MIN_EC_DEBATE_ROUNDS 미만이면 finalized 무시
    if state["iteration"] > 0:
        if state["iteration"] + 1 >= MIN_EC_DEBATE_ROUNDS:
            # 복합 EC / EC 수 범위 위반이면 finalized를 거부 (미커버 CR은 기록만 — 발송 10 A1)
            if compound_warnings:
                print(f"    ⚠️  복합 EC {len(compound_warnings)}개 발견 — finalized 거부, 추가 라운드")
                update["consensus"] = False
            elif ec_out_of_range:
                print(f"    ⚠️  EC 수 {ec_count}개 (허용: {min_ecs}–{max_ecs}) — finalized 거부, 추가 라운드")
                update["consensus"] = False
            else:
                # 발송 10 A1: 미커버 CR은 합의를 막지 않고 기록만 (EC 상한 우선)
                if gaps:
                    print(f"    ℹ️  미커버 CR {len(gaps)}개 — 기록 (EC 상한 우선, 합의 판정에 영향 없음)")
                update["consensus"] = result.get("finalized", False)
        else:
            update["consensus"] = False

    return update


def market_expert_ec_node(state: QFDState) -> dict:
    """Market Expert: EC 리스트 검토 (도구 호출 가능)"""
    seg_a = state["segment_a"]
    seg_b = state["segment_b"]
    cr_all = _dedupe_crs(state["cr_list_a"], state["cr_list_b"])
    cr_ids = _make_id_map([c["cr"] for c in cr_all], "CR")
    cr_name_to_id = {v: k for k, v in cr_ids.items()}
    cr_text_a = _format_crs_with_ids(state["cr_list_a"], cr_name_to_id)
    cr_text_b = _format_crs_with_ids(state["cr_list_b"], cr_name_to_id)
    min_ecs, max_ecs = _compute_ec_range(state)

    ec_text = "\n".join(
        f'  EC-{i}: "{ec["ec"]}"  ← covers: '
        + (", ".join(cr_name_to_id.get(c, "?") for c in ec.get("covers", [])) or "(none)")
        for i, ec in enumerate(state["ec_list"], 1)
    )
    gaps = state.get("coverage_gaps", []) or []
    uncovered_text = (
        "### CRs covered by NO EC (empty rows — Engineering's own mapping):\n"
        + "\n".join(f'  - {cr_name_to_id.get(g, "?")}: "{g}"' for g in gaps) + "\n"
        if gaps else
        "### CRs covered by NO EC: none (every CR has at least one EC)\n"
    )

    is_first_turn = not any(m["agent"] == "market_expert" for m in state["messages"])

    tool_instruction = (
        "FIRST: You MUST use your search tools to gather market evidence before responding.\n"
        "Search for specific consumer topics relevant to the CRs (e.g., 'laptop battery life\n"
        "consumer satisfaction', 'laptop weight portability user preference') — NOT generic queries.\n\n"
        if is_first_turn else
        "You may use search tools if you need additional evidence.\n\n"
    )

    user_content = f"""Review the Engineering Expert's unified EC proposal.
This EC set is shared across ALL segments — do NOT suggest segment-specific ECs.

### {seg_a} Customer Requirements ({len(state['cr_list_a'])}):
{cr_text_a}

### {seg_b} Customer Requirements ({len(state['cr_list_b'])}):
{cr_text_b}

### Proposed ECs ({len(state['ec_list'])}, allowed range: {min_ecs}–{max_ecs}):
{ec_text}

{uncovered_text}
{tool_instruction}THEN: Evaluate the EC set on these criteria:
1. **CR coverage**: Are both segments' key CRs addressable by the proposed ECs?
2. **EC quality**: Flag any EC that is COMPOUND (multiple attributes in one),
   non-engineering (UX/software/planning), or too vague to measure.
3. **Redundancy**: Are any ECs measuring essentially the same engineering trade-off?
   If so, recommend merging them. Fewer well-chosen ECs are better than many overlapping ones.
4. **Missing coverage (empty rows)**: For every CR listed above as covered by NO EC,
   name it by ID in "missing_aspects" and suggest a measurable EC for it (and which
   existing EC to merge or drop to make room). Also check whether any "covers" claim
   looks unjustified.

Focus on the 3–5 most impactful issues.
If suggesting a new EC, also suggest which existing EC to replace or merge.
Every suggestion MUST cite evidence from your tool results or the CRs.
ONLY suggest measurable engineering design parameters as ECs.

Output JSON:
{{
    "feedback_summary": "1-2 sentence overall assessment",
    "ec_feedback": [{{"ec": "name", "issue": "description", "suggestion": "proposed change", "evidence": "specific data from tool results or reviews"}}],
    "cr_coverage": "assessment of how well CRs from both segments are covered",
    "missing_aspects": ["CR not covered — suggest which engineering EC to add and which to replace"]
}}"""

    llm_with_tools = _chat_llm.bind_tools(MARKET_TOOLS)
    messages = [
        {"role": "system", "content": MARKET_EXPERT_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 6

    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [mkt] {tool_name}({list(tool_args.values())[0][:60]}...)")

            tool_map = {t.name: t for t in MARKET_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [mkt] 도구 {tool_call_count}회 호출 완료")

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(
            MARKET_EXPERT_SYSTEM,
            user_content + "\n\nIMPORTANT: Output ONLY valid JSON. No other text.",
        )

    return {
        "messages": [
            {
                "agent": "market_expert",
                "round": state["iteration"],
                "content": result,
            }
        ],
    }


def ec_consensus_check(state: QFDState) -> str:
    """EC 토론 합의/라운드 체크"""
    if state["iteration"] >= settings.MAX_DEBATE_ROUNDS or state.get("consensus", False):
        return "end"
    return "continue"


def build_ec_debate_graph():
    """Phase 2-A: EC 리스트 토론 그래프"""
    graph = StateGraph(QFDState)

    graph.add_node("engineering_expert", engineering_expert_ec_node)
    graph.add_node("market_expert", market_expert_ec_node)

    graph.set_entry_point("engineering_expert")
    graph.add_conditional_edges("engineering_expert", ec_consensus_check, {
        "continue": "market_expert",
        "end": END,
    })
    graph.add_edge("market_expert", "engineering_expert")

    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Roof Debate (Phase 2-B) — Roof Matrix 토론
# 확정된 EC에 대해 EC 간 상관관계만 토론
# ═════════════════════════════════════════════════════════════════════════════

def _generate_ec_pairs(ec_list: list[dict]) -> str:
    """EC 리스트에서 모든 고유 쌍 목록을 생성"""
    ec_names = [ec["ec"] for ec in ec_list]
    lines = []
    idx = 1
    for i in range(len(ec_names)):
        for j in range(i + 1, len(ec_names)):
            lines.append(f"  {idx}. EC-{i + 1} ↔ EC-{j + 1}")
            idx += 1
    return "\n".join(lines)


def engineering_expert_roof_node(state: QFDState) -> dict:
    """Engineering Expert: Roof Matrix 제안/수정"""
    ec_text = _format_ecs_ids(state["ec_list"])
    ec_ids = _make_id_map([ec["ec"] for ec in state["ec_list"]], "EC")
    ec_pairs_text = _generate_ec_pairs(state["ec_list"])
    n_ecs = len(state["ec_list"])
    n_pairs = n_ecs * (n_ecs - 1) // 2

    if state["iteration"] == 0:
        user_content = f"""Build the Roof Matrix (EC × EC correlation) for the confirmed EC set.

### Confirmed ECs ({n_ecs}):
{ec_text}

### All EC Pairs to Evaluate ({n_pairs} pairs):
{ec_pairs_text}

For EACH pair above, decide:
- If there is a meaningful physical/engineering correlation → assign a value:
  -0.9 (strong negative) | -0.5 (weak negative) | 0.5 (weak positive) | 0.9 (strong positive)
- If there is genuinely NO physical coupling → omit the pair.

You may use search tools to verify technical correlations if needed.

Use the EC IDs shown above (e.g. "EC-3") as JSON keys.
IMPORTANT: Answer with IDs ONLY — do NOT retype EC names.

Output JSON:
{{
    "roof_matrix": {{"<EC id>": {{"<EC id>": <-0.9|-0.5|0.5|0.9>}}}},
    "reasoning": "Explanation of key correlations and their engineering basis"
}}"""
    else:
        feedback = [m for m in state["messages"] if m["agent"] == "market_expert"][-1]
        cur_roof = state.get("roof_matrix", {})
        name_to_id = {v: k for k, v in ec_ids.items()}
        cur_roof_text = _format_roof_lines(cur_roof, name_to_id)
        n_cur = sum(len(r) for r in cur_roof.values() if isinstance(r, dict))
        user_content = f"""The Market Expert reviewed your Roof Matrix:

{json.dumps(feedback["content"], indent=2, ensure_ascii=False)}

### Your current Roof Matrix ({n_cur} entries):
{cur_roof_text}

Revise THIS matrix: keep every pair you do not explicitly change, and add or
modify only the pairs raised in the feedback (or that you now judge wrong).
If you remove a pair, say why in feedback_responses. Output the FULL matrix.

Address each concern. Use search tools for additional evidence if needed.

Allowed values ONLY: {{-0.9, -0.5, 0.5, 0.9}}

### ECs ({n_ecs}):
{ec_text}

Use the EC IDs above (e.g. "EC-3") as JSON keys.
IMPORTANT: Answer with IDs ONLY — do NOT retype EC names.

Output JSON:
{{
    "roof_matrix": {{"<EC id>": {{"<EC id>": <-0.9|-0.5|0.5|0.9>}}}},
    "feedback_responses": [{{"concern": "original concern", "action": "ACCEPT|REJECT", "detail": "what changed or why rejected"}}],
    "reasoning": "Summary of changes",
    "finalized": true | false
}}"""

    llm_with_tools = _chat_llm.bind_tools(ENGINEERING_TOOLS)
    messages = [
        {"role": "system", "content": ENGINEERING_EXPERT_ROOF_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4

    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [eng/roof] {tool_name}({list(tool_args.values())[0][:60]}...)")

            tool_map = {t.name: t for t in ENGINEERING_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(
            ENGINEERING_EXPERT_ROOF_SYSTEM,
            user_content + "\n\nIMPORTANT: Output ONLY valid JSON.",
        )

    result = _validate_roof(result)

    # ID → 정본 EC 이름 치환 + 키 검증 (해석 불가 키는 경고 후 폐기)
    resolved_roof = _resolve_matrix_ids(
        result.get("roof_matrix", {}), ec_ids, ec_ids, "Roof"
    )
    result["roof_matrix"] = resolved_roof
    print(f"    [Roof] EC 키 확정: {len(resolved_roof)}개 행 (정본 대조 완료)")

    update = {
        "roof_matrix": result.get("roof_matrix", {}),
        "iteration": state["iteration"] + 1,
        "messages": [
            {
                "agent": "engineering_expert",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }

    if state["iteration"] > 0:
        if state["iteration"] + 1 >= MIN_ROOF_DEBATE_ROUNDS:
            update["consensus"] = result.get("finalized", False)
        else:
            update["consensus"] = False

    return update


def market_expert_roof_node(state: QFDState) -> dict:
    """Market Expert: Roof Matrix 검토"""
    ec_text = _format_ecs_ids(state["ec_list"])
    ec_pairs_text = _generate_ec_pairs(state["ec_list"])
    n_ecs = len(state["ec_list"])
    n_pairs = n_ecs * (n_ecs - 1) // 2

    # 현재 Roof에 존재하는 쌍 수 계산
    existing_pairs = set()
    for a in state["roof_matrix"]:
        for b in state["roof_matrix"][a]:
            existing_pairs.add(tuple(sorted([a, b])))

    user_content = f"""Review the Roof Matrix for accuracy.

### ECs ({n_ecs}):
{ec_text}

### All Possible EC Pairs ({n_pairs} total):
{ec_pairs_text}

### Current Roof Matrix ({len(existing_pairs)} pairs assigned out of {n_pairs}):
{json.dumps(state["roof_matrix"], indent=2, ensure_ascii=False)}

### Review Checklist:
1. **Missing correlations**: Cross-check the full pair list above against the
   current matrix. Flag any pair that should have a correlation but is missing.
2. **Incorrect signs**: For each existing correlation, verify the sign matches
   the physical co-movement (EC_A value UP → EC_B value UP = positive).
3. **Magnitude appropriateness**: Is -0.9 vs -0.5 justified?
4. **Spurious correlations**: Any pair that has a value but shouldn't?

Focus on the most impactful issues. Cite evidence.

Output JSON:
{{
    "feedback_summary": "1-2 sentence assessment",
    "coverage_assessment": "X out of {n_pairs} pairs evaluated — are key pairs missing?",
    "roof_concerns": [{{"ec_a": "<EC id>", "ec_b": "<EC id>", "issue": "description", "evidence": "data source"}}]
}}"""

    llm_with_tools = _chat_llm.bind_tools(MARKET_TOOLS)
    messages = [
        {"role": "system", "content": MARKET_EXPERT_ROOF_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4

    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [mkt/roof] {tool_name}({list(tool_args.values())[0][:60]}...)")

            tool_map = {t.name: t for t in MARKET_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(
            MARKET_EXPERT_ROOF_SYSTEM,
            user_content + "\n\nIMPORTANT: Output ONLY valid JSON.",
        )

    return {
        "messages": [
            {
                "agent": "market_expert",
                "round": state["iteration"],
                "content": result,
            }
        ],
    }


def roof_consensus_check(state: QFDState) -> str:
    """Roof 토론 합의/라운드 체크"""
    if state["iteration"] >= settings.MAX_DEBATE_ROUNDS or state.get("consensus", False):
        return "end"
    return "continue"


def build_roof_debate_graph():
    """Phase 2-B: Roof Matrix 토론 그래프"""
    graph = StateGraph(QFDState)

    graph.add_node("engineering_expert", engineering_expert_roof_node)
    graph.add_node("market_expert", market_expert_roof_node)

    graph.set_entry_point("engineering_expert")
    graph.add_conditional_edges("engineering_expert", roof_consensus_check, {
        "continue": "market_expert",
        "end": END,
    })
    graph.add_edge("market_expert", "engineering_expert")

    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Combined HoQ Debate (Phase 2-A + 2-B 순차 실행)
# 하위 호환: build_hoq_debate_graph()는 EC → Roof 순차 실행 래퍼
# ═════════════════════════════════════════════════════════════════════════════

def run_hoq_debate(state: QFDState) -> dict:
    """Phase 2-A (EC 토론) + Phase 2-B (Roof 토론) 순차 실행

    Returns:
        ec_list, roof_matrix, iteration, consensus, messages 를 포함한 결과 dict
    """
    # ── Phase 2-A: EC 토론 ──
    print(f"\n  [Phase 2-A] EC 리스트 토론 시작")
    ec_graph = build_ec_debate_graph()
    ec_state: QFDState = {
        "segment_a": state["segment_a"],
        "segment_b": state["segment_b"],
        "cr_list_a": state["cr_list_a"],
        "cr_list_b": state["cr_list_b"],
        "phase": "ec",
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }
    ec_result = ec_graph.invoke(ec_state)

    ec_rounds = ec_result["iteration"]
    ec_messages = ec_result["messages"]

    print(f"  [Phase 2-A] EC 토론 완료: {ec_rounds}라운드, EC {len(ec_result['ec_list'])}개")

    # ── Phase 2-B: Roof 토론 ──
    print(f"\n  [Phase 2-B] Roof Matrix 토론 시작")
    roof_graph = build_roof_debate_graph()
    roof_state: QFDState = {
        "segment_a": state["segment_a"],
        "segment_b": state["segment_b"],
        "cr_list_a": state["cr_list_a"],
        "cr_list_b": state["cr_list_b"],
        "ec_list": ec_result["ec_list"],
        "phase": "roof",
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }
    roof_result = roof_graph.invoke(roof_state)

    roof_rounds = roof_result["iteration"]
    roof_messages = roof_result["messages"]
    print(f"  [Phase 2-B] Roof 토론 완료: {roof_rounds}라운드")

    # 토론 로그 저장
    save_debate_log(
        "stage1_ec_debate.txt",
        ec_messages,
        f"Stage 1 — EC List Debate ({ec_rounds} rounds, {len(ec_result['ec_list'])} ECs)",
    )
    save_debate_log(
        "stage1_roof_debate.txt",
        roof_messages,
        f"Stage 1 — Roof Matrix Debate ({roof_rounds} rounds)",
    )

    # 결합 — consensus는 하드코딩하지 않고 각 토론의 실제 합의 여부를 기록
    ec_consensus = ec_result.get("consensus", False)
    roof_consensus = roof_result.get("consensus", False)
    return {
        "ec_list": ec_result["ec_list"],
        "roof_matrix": roof_result["roof_matrix"],
        "iteration": ec_rounds + roof_rounds,
        "consensus": ec_consensus and roof_consensus,
        "ec_consensus": ec_consensus,
        "roof_consensus": roof_consensus,
        "messages": [
            *[{**m, "phase": "ec"} for m in ec_messages],
            *[{**m, "phase": "roof"} for m in roof_messages],
        ],
        "ec_rounds": ec_rounds,
        "roof_rounds": roof_rounds,
        # 발송 9 #22: EC 토론 종료 시점의 미커버 CR (마지막 라운드까지 남은 빈 행)
        "ec_coverage_gaps": ec_result.get("coverage_gaps", []) or [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Per-segment Matrix Fill (Debate-based)
# Engineering Expert ↔ Market Expert 토론으로 Relationship Matrix 채움
# 각 Agent는 2-Phase CoT (Reasoning → JSON) 방식으로 추론
# ═════════════════════════════════════════════════════════════════════════════

MATRIX_FILL_CHUNK_SIZE = 5  # CR 분할 단위

_PRIORITY_SUFFIX_RE = __import__("re").compile(r"\s*\(priority:\s*[\d.]+\)\s*$")


def _normalize_cr_keys(
    rel_matrix: dict, cr_list: list[dict]
) -> dict:
    """direction/importance matrix 키에서 '(priority: X.X)' suffix를 제거하고
    cr_list의 정규 CR 라벨로 매핑."""
    canonical = {cr["cr"] for cr in cr_list}
    normalized: dict = {}
    for key, value in rel_matrix.items():
        if key in canonical:
            normalized[key] = value
            continue
        clean = _PRIORITY_SUFFIX_RE.sub("", key).strip()
        if clean in canonical:
            normalized[clean] = value
        else:
            normalized[clean] = value
    return normalized


def _format_crs_numbered(cr_list: list[dict]) -> str:
    """CR을 번호 + 텍스트 + priority로 포맷 (키와 priority를 시각적으로 분리)"""
    lines = []
    for i, cr in enumerate(cr_list, 1):
        lines.append(f'{i}. "{cr["cr"]}"')
        lines.append(f"   priority: {cr['priority']}")
    return "\n".join(lines)


_TI_SUFFIX_RE = __import__("re").compile(r"\s*\(TI:\s*[\d.]+\)\s*$")


def _format_ecs_numbered(
    ec_list: list[dict], technical_importance: dict[str, float] | None = None
) -> str:
    """EC를 번호 + 텍스트 + TI로 포맷 (키와 TI를 시각적으로 분리)

    EC 라벨 자체가 괄호를 여러 겹 포함하는 긴 문장이므로 `EC (TI: 45.0)` 처럼
    한 줄에 붙이면 LLM이 주석 경계를 찾지 못해 TI까지 키에 포함시킨다.
    _format_crs_numbered와 동일한 방식으로 줄을 나눈다.
    """
    lines = []
    for i, ec in enumerate(ec_list, 1):
        lines.append(f'{i}. "{ec["ec"]}"')
        if technical_importance is not None:
            lines.append(f'   TI: {technical_importance.get(ec["ec"], 0):.1f}')
    return "\n".join(lines)


def _normalize_ec_keys(rel_matrix: dict, ec_list: list[dict]) -> dict:
    """relationship matrix 키에서 '(TI: X.X)' suffix를 제거하고
    ec_list의 정규 EC 라벨로 매핑."""
    canonical = {ec["ec"] for ec in ec_list}
    normalized: dict = {}
    for key, value in rel_matrix.items():
        if key in canonical:
            normalized[key] = value
            continue
        clean = _TI_SUFFIX_RE.sub("", key).strip()
        if clean != key and clean in canonical:
            print(f"    ⚠️  [정규화] EC 키에서 TI suffix 제거: {clean[:55]}...")
        normalized[clean] = value
    return normalized


# ─── 번호 기반 키 (ID) 헬퍼 ─────────────────────────────────────────────────
# LLM에게 긴 CR/EC/PC 이름을 JSON 키로 다시 타이핑하게 하지 않는다.
# 프롬프트에는 'CR-3: "..."' 형식으로 ID를 붙여 제시하고, LLM은 ID로만 답하며,
# 코드가 ID를 저장된 정본 이름으로 치환한다.

def _format_crs_ids(cr_list: list[dict], start: int = 1) -> str:
    """CR을 'CR-n' ID + 텍스트 + priority로 포맷"""
    lines = []
    for i, cr in enumerate(cr_list, start):
        lines.append(f'CR-{i}: "{cr["cr"]}"')
        lines.append(f"   priority: {cr['priority']}")
    return "\n".join(lines)


def _format_ecs_ids(
    ec_list: list[dict],
    technical_importance: dict[str, float] | None = None,
    start: int = 1,
) -> str:
    """EC를 'EC-n' ID + 텍스트(+TI)로 포맷"""
    lines = []
    for i, ec in enumerate(ec_list, start):
        lines.append(f'EC-{i}: "{ec["ec"]}"')
        if technical_importance is not None:
            lines.append(f'   TI: {technical_importance.get(ec["ec"], 0):.1f}')
    return "\n".join(lines)


def _format_roof_lines(roof: dict, name_to_id: dict[str, str]) -> str:
    """roof 행렬을 'ID ↔ ID: 값' 한 줄 표기로 압축 (라운드 프롬프트용, JSON보다 짧음)"""
    lines = []
    seen: set[tuple[str, str]] = set()
    for a, row in roof.items():
        if not isinstance(row, dict):
            continue
        for b, v in row.items():
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            ia = name_to_id.get(a, a)
            ib = name_to_id.get(b, b)
            lines.append(f"  {ia} ↔ {ib}: {v}")
    return "\n".join(lines) if lines else "  (none)"


def _make_id_map(names: list[str], prefix: str, start: int = 1) -> dict[str, str]:
    """{'EC-1': <정본 이름>, ...} — ID → 정본 매핑"""
    return {f"{prefix}-{i}": name for i, name in enumerate(names, start)}


def _resolve_key(
    key: str, id_map: dict[str, str], canonical: set[str], context: str
) -> str | None:
    """LLM 응답 키를 정본 이름으로 치환.

    ID 일치 → 정본 이름 반환. ID가 아니어도 정본 이름과 정확히 일치하면 통과.
    suffix 제거로 복구 가능하면 경고 후 복구, 그 외에는 경고 후 None(폐기).
    조용히 통과시키지 않는다."""
    k = str(key).strip()
    if k in id_map:
        return id_map[k]
    if k in canonical:
        return k
    stripped = _TI_SUFFIX_RE.sub("", _PRIORITY_SUFFIX_RE.sub("", k)).strip()
    if stripped in canonical:
        print(f"    ⚠️  [{context}] suffix 제거로 복구된 키: {stripped[:55]}...")
        return stripped
    print(f"    ⚠️  [{context}] 해석 불가 키 폐기: {k[:70]!r}")
    return None


def _resolve_matrix_ids(
    matrix: dict,
    outer_ids: dict[str, str],
    inner_ids: dict[str, str],
    context: str,
) -> dict:
    """{외곽 ID: {내부 ID: 값}} 형태 행렬의 키를 정본 이름으로 치환"""
    outer_canonical = set(outer_ids.values())
    inner_canonical = set(inner_ids.values())
    resolved: dict = {}
    for okey, row in matrix.items():
        oname = _resolve_key(okey, outer_ids, outer_canonical, context)
        if oname is None:
            continue
        if not isinstance(row, dict):
            resolved[oname] = row
            continue
        inner: dict = {}
        for ikey, val in row.items():
            iname = _resolve_key(ikey, inner_ids, inner_canonical, context)
            if iname is None:
                continue
            inner[iname] = val
        resolved[oname] = inner
    return resolved


def _build_prev_context(accumulated_rel: dict, ec_list: list[dict]) -> str:
    """이전 Chunk 결과를 EC별 패턴 요약으로 변환 (맥락 누적용)"""
    if not accumulated_rel:
        return ""

    ec_names = [ec["ec"] for ec in ec_list]
    lines = ["### Previously Assessed CR-EC Relationships (for sign/magnitude consistency):"]

    for ec_name in ec_names:
        ec_entries = []
        for cr, ecs in accumulated_rel.items():
            val = ecs.get(ec_name, 0)
            if val != 0:
                short_cr = cr[:60] + "..." if len(cr) > 60 else cr
                ec_entries.append(f'    "{short_cr}" → {val:+d}')
        if ec_entries:
            lines.append(f'\nEC "{ec_name}":')
            lines.extend(ec_entries)

    lines.append(
        "\n⚠️ Maintain sign consistency: CRs wanting the SAME physical outcome "
        "from the same EC should have the SAME sign as above."
    )
    return "\n".join(lines)


def fill_segment_matrix(
    ec_list: list[dict],
    cr_list: list[dict],
    segment: str | None = None,
) -> dict:
    """Engineering Expert ↔ Market Expert 토론으로 Relationship Matrix를 채움.

    각 Chunk (5 CRs)마다 최대 MAX_DEBATE_ROUNDS 라운드 토론:
      Round N:
        1) Engineering Expert  Phase 1 (CoT) → Phase 2 (JSON proposal)
        2) Market Expert       Phase 1 (CoT) → Phase 2 (JSON verdict)
      Market approved → 다음 Chunk / MAX_ROUNDS 도달 → 마지막 제안 채택

    Args:
        ec_list: 공유 EC 리스트 [{"ec": str, ...}]
        cr_list: CR 리스트 [{"cr": str, "priority": float}]
        segment: "Professional" | "Casual" | None (Common CRs)

    Returns:
        {"relationship_matrix": {...}, "importance_matrix": {...}, "direction_matrix": {...}}
    """
    fill_label = segment or "Common"
    ec_text = _format_ecs_ids(ec_list)
    ec_ids = _make_id_map([ec["ec"] for ec in ec_list], "EC")
    max_rounds = settings.MAX_DEBATE_ROUNDS

    # ── 리뷰 데이터 사전 수집 ──
    if segment is None:
        block_label = "Common (shared across all user segments)"
        review_parts = []
        for seg in [settings.SEGMENT_A, settings.SEGMENT_B]:
            rev = _retrieve_reviews(seg, cr_list)
            if rev:
                review_parts.append(f"### {seg} user reviews:\n{rev}")
        review_section = "\n\n".join(review_parts) + "\n" if review_parts else ""
    else:
        block_label = f'"{segment}" user segment'
        review_text = _retrieve_reviews(segment, cr_list)
        review_section = (
            f"### Raw Review Data (reference):\n{review_text}\n"
            if review_text
            else ""
        )

    segment_instruction = (
        "These are Common CRs shared by all segments — evaluate the "
        "technical relationship objectively."
        if segment is None
        else f'Consider the "{segment}" user\'s specific usage context.'
    )

    # ── CR Chunk 분할 ──
    chunks = [
        cr_list[i : i + MATRIX_FILL_CHUNK_SIZE]
        for i in range(0, len(cr_list), MATRIX_FILL_CHUNK_SIZE)
    ]
    total_chunks = len(chunks)
    print(
        f"  [{fill_label}] Matrix Fill 토론 시작 — "
        f"{len(cr_list)} CRs ÷ {MATRIX_FILL_CHUNK_SIZE} = {total_chunks} chunks, "
        f"max {max_rounds} rounds/chunk"
    )

    accumulated_rel: dict = {}
    fill_log_messages: list[dict] = []
    total_debate_rounds = 0

    for chunk_idx, chunk_crs in enumerate(chunks, 1):
        chunk_start = (chunk_idx - 1) * MATRIX_FILL_CHUNK_SIZE + 1
        cr_text = _format_crs_ids(chunk_crs, start=chunk_start)
        cr_ids = _make_id_map(
            [cr["cr"] for cr in chunk_crs], "CR", start=chunk_start
        )
        prev_context = _build_prev_context(accumulated_rel, ec_list)

        chunk_consensus = False
        eng_proposal: dict = {}
        eng_reasoning_text = ""
        mkt_reasoning_text = ""
        mkt_adjustments: list[dict] = []

        for debate_round in range(1, max_rounds + 1):
            total_debate_rounds += 1

            # ══════════════════════════════════════════════════
            # Engineering Expert — Phase 1: CoT Reasoning
            # ══════════════════════════════════════════════════
            if debate_round == 1:
                eng_user_msg = f"""Analyze the CR-EC relationships for {block_label} CRs.
(Chunk {chunk_idx}/{total_chunks})

### Engineering Characteristics (ECs) — fixed, shared across segments:
{ec_text}

### Customer Requirements (CRs) to analyze:
{cr_text}

{review_section}
{prev_context}

### Task:
{segment_instruction}

For EACH CR, analyze its relationship with EVERY EC:
1. What does this CR physically require?
2. For each EC: When the EC value goes UP, is this CR better (+) or worse (-)? Why?
3. How strong is the link — primary driver (strong ±9) or secondary factor (moderate ±3)?

Think step by step. Be specific about the causal chain for each CR-EC pair."""
            else:
                eng_user_msg = f"""The Market Expert reviewed your proposal and provided feedback.

### Market Expert's reasoning:
{mkt_reasoning_text}

### Market Expert's suggested adjustments:
{json.dumps(mkt_adjustments, indent=2, ensure_ascii=False)}

### Your previous proposal:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### CRs:
{cr_text}

### ECs:
{ec_text}

{prev_context}

### Task:
Address EACH adjustment from the Market Expert with ACCEPT or REJECT:
- ACCEPT: Explain why, and state the revised value.
- REJECT: Explain with engineering reasoning why the original value is correct.

After addressing all feedback, provide your revised analysis for the full chunk."""

            print(
                f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} "
                f"Round {debate_round}/{max_rounds} — Eng. Expert CoT..."
            )
            eng_reasoning_text = _llm_text(
                build_matrix_fill_reasoning_system("engineering"), eng_user_msg
            )

            fill_log_messages.append({
                "agent": "engineering_expert_reasoning",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": eng_reasoning_text,
            })

            # ══════════════════════════════════════════════════
            # Engineering Expert — Phase 2: JSON Structuring
            # ══════════════════════════════════════════════════
            eng_json_msg = f"""Convert your reasoning into a Relationship Matrix.

### Your reasoning analysis:
{eng_reasoning_text}

### CRs in this chunk:
{cr_text}

### ECs:
{ec_text}

### Rules:
- Follow your reasoning analysis faithfully.
- Allowed values ONLY: {{-9, -3, 0, 3, 9}}
- Use the CR IDs and EC IDs shown above (e.g. "CR-3", "EC-7") as JSON keys.
  IMPORTANT: Answer with IDs ONLY — do NOT retype CR or EC names.
- Every CR-EC pair MUST have a value (use 0 if no connection).

Output JSON:
{{
    "relationship_matrix": {{"<CR id>": {{"<EC id>": <-9|-3|0|3|9>}}}}
}}"""

            print(
                f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} "
                f"Round {debate_round}/{max_rounds} — Eng. Expert JSON..."
            )
            eng_result = _llm_json(build_matrix_fill_json_system("engineering"), eng_json_msg)
            eng_result = _validate_relationship(eng_result)
            eng_proposal = _resolve_matrix_ids(
                eng_result.get("relationship_matrix", {}),
                cr_ids, ec_ids, f"{fill_label} MatrixFill",
            )

            fill_log_messages.append({
                "agent": "engineering_expert_json",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": eng_proposal,
            })

            # ══════════════════════════════════════════════════
            # Market Expert — Phase 1: CoT Review
            # ══════════════════════════════════════════════════
            mkt_user_msg = f"""Review the Engineering Expert's CR-EC relationship proposal.

### Engineering Expert's reasoning:
{eng_reasoning_text}

### Engineering Expert's proposed values:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### CRs:
{cr_text}

### ECs:
{ec_text}

{review_section}

### Task:
{segment_instruction}

Evaluate the proposal from a user/market perspective:
1. For each CR-EC pair: Is the sign correct? Is the magnitude appropriate?
2. Are there relationships rated 0 that real users would strongly feel?
3. Are there values rated ±9 that are actually secondary factors?
4. Does the proposal match user complaints and priorities in the review data?

Be specific about which values you agree or disagree with, and why."""

            print(
                f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} "
                f"Round {debate_round}/{max_rounds} — Mkt. Expert CoT..."
            )
            mkt_reasoning_text = _llm_text(
                build_matrix_fill_reasoning_system("market"), mkt_user_msg
            )

            fill_log_messages.append({
                "agent": "market_expert_reasoning",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": mkt_reasoning_text,
            })

            # ══════════════════════════════════════════════════
            # Market Expert — Phase 2: JSON Verdict
            # ══════════════════════════════════════════════════
            mkt_json_msg = f"""Structure your review into a JSON verdict.

### Your review analysis:
{mkt_reasoning_text}

### Engineering Expert's current proposal:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### CRs:
{cr_text}

### Rules:
- Set "approved": true if the proposal is acceptable overall.
- Set "approved": false if there are significant values that need change.
- Only list values you genuinely disagree with in "adjustments".
- Use the CR ID (e.g. "CR-3") for the "cr" field and the EC ID (e.g. "EC-7")
  for the "ec" field — do NOT retype names.
- Suggested values must be in {{-9, -3, 0, 3, 9}}.

Output JSON:
{{
    "approved": true/false,
    "overall_assessment": "1-2 sentence summary",
    "adjustments": [
        {{"cr": "<CR id>", "ec": "<EC id>", "current": <value>, "suggested": <value>, "reason": "brief justification"}}
    ]
}}"""

            print(
                f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} "
                f"Round {debate_round}/{max_rounds} — Mkt. Expert JSON..."
            )
            mkt_result = _llm_json(build_matrix_fill_json_system("market"), mkt_json_msg)
            mkt_adjustments = mkt_result.get("adjustments", [])
            chunk_consensus = mkt_result.get("approved", False)

            fill_log_messages.append({
                "agent": "market_expert_json",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": mkt_result,
            })

            adj_count = len(mkt_adjustments)
            status = "✅ approved" if chunk_consensus else f"❌ {adj_count} adjustments"
            print(
                f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} "
                f"Round {debate_round}/{max_rounds} — {status}"
            )

            if chunk_consensus:
                break

        # ── Chunk 결과 누적 ──
        accumulated_rel.update(eng_proposal)
        print(
            f"    [{fill_label}] Chunk {chunk_idx}/{total_chunks} 완료 — "
            f"{len(eng_proposal)} CRs, {debate_round} rounds"
        )

    # ── CR 키 정합 확인 + Importance / Direction 도출 ──
    # (ID 치환이 라운드마다 수행되므로 여기서는 정본 일치만 확인)
    canonical_crs = {cr["cr"] for cr in cr_list}
    matched_crs = sum(1 for k in accumulated_rel if k in canonical_crs)
    print(f"  [{fill_label}] CR 키 일치: {matched_crs}/{len(cr_list)}")
    if matched_crs < len(cr_list):
        print(f"  ⚠️  [{fill_label}] {len(cr_list) - matched_crs}개 CR이 "
              f"행렬에서 누락되었습니다.")
    importance = {
        cr: {ec: abs(v) for ec, v in ecs.items()}
        for cr, ecs in accumulated_rel.items()
    }
    direction = {
        cr: {ec: (1 if v > 0 else -1 if v < 0 else 0) for ec, v in ecs.items()}
        for cr, ecs in accumulated_rel.items()
    }

    print(
        f"  [{fill_label}] Matrix Fill 토론 완료 — "
        f"{len(accumulated_rel)} CRs x {len(ec_list)} ECs, "
        f"총 {total_debate_rounds} debate rounds"
    )
    # 10/11: 밀도는 규칙이 아니라 관찰값으로만 기록 (비0 비율, CR당 평균 연결 EC 수)
    _cells = [v for row in accumulated_rel.values() for v in row.values()]
    _nz = sum(1 for v in _cells if v)
    if _cells:
        print(f"  [{fill_label}] 관찰: 비0 {_nz}/{len(_cells)} ({100*_nz/len(_cells):.0f}%), "
              f"CR당 평균 {_nz/max(len(accumulated_rel),1):.1f} ECs")

    save_debate_log(
        f"stage1_matrix_fill_{fill_label.lower().replace(' ', '_')}.txt",
        fill_log_messages,
        f"Stage 1 — Matrix Fill Debate [{fill_label}] "
        f"({len(cr_list)} CRs, {total_chunks} chunks, "
        f"{total_debate_rounds} total rounds)",
    )

    return {
        "relationship_matrix": accumulated_rel,
        "importance_matrix": importance,
        "direction_matrix": direction,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Conflict Resolution Debate
# Phase A: Advocate Debate
#   orchestrator → pro_advocate → cas_advocate
#        ↑________________________________| (continue)
#                                         ↓ (end) → debate summary 반환
# Phase B: Judge (2-Agent)
#   market_expert_judge ↔ engineering_expert_judge
#        → strategy + adjusted importance 반환
# ═════════════════════════════════════════════════════════════════════════════

# Advocate에게 부여할 도구: ChromaDB 리뷰 검색 전용
ADVOCATE_TOOLS = [search_reviews]
# Judge(Market)에게 부여할 도구
JUDGE_MARKET_TOOLS = [search_tavily, search_google, search_reviews]
# Judge(Engineering)에게 부여할 도구
JUDGE_ENGINEERING_TOOLS = [search_arxiv, search_google, search_reviews]


def _build_conflict_context(state: QFDState) -> str:
    """현재 토론 중인 충돌 그룹의 컨텍스트 문자열 생성"""
    ctx = state["current_conflict"]
    lines = ["### Conflict Group Context"]

    # 충돌 CR pairs
    lines.append(f"\n**Segment A ({state['segment_a']}) CRs in this group:**")
    for cr in ctx.get("cr_a_list", []):
        cr_data = next((c for c in state["cr_list_a"] if c["cr"] == cr), {})
        priority = cr_data.get("priority", "?")
        lines.append(f'  - "{cr}" (priority: {priority})')

    lines.append(f"\n**Segment B ({state['segment_b']}) CRs in this group:**")
    for cr in ctx.get("cr_b_list", []):
        cr_data = next((c for c in state["cr_list_b"] if c["cr"] == cr), {})
        priority = cr_data.get("priority", "?")
        lines.append(f'  - "{cr}" (priority: {priority})')

    if ctx.get("cr_common_list"):
        lines.append("\n**Common CRs in this group:**")
        all_cr_dicts = (
            state.get("cr_list_common", [])
            + state.get("cr_list_a", [])
            + state.get("cr_list_b", [])
        )
        for cr in ctx["cr_common_list"]:
            cr_data = next((c for c in all_cr_dicts if c["cr"] == cr), {})
            priority = cr_data.get("priority", "?")
            lines.append(f'  - "{cr}" (priority: {priority})')

    # 10/13: 이 그룹의 충돌 셀(= Judge의 Adjustable cells)을 토론에도 동일하게 제시 —
    # 양보·양보 한계는 이 칸들을 두고 이뤄져야 조정으로 이어진다.
    scope = _group_scope_cells(state)
    if scope:
        imp_now = state.get("importance_matrix", {})
        lines.append("\n**Contested cells of this group (concessions count only here):**")
        for cr, ec in sorted(scope):
            lines.append(f'  - "{cr[:60]}..." × "{ec}": current importance '
                         f'{imp_now.get(cr, {}).get(ec, 0)}')

    # 충돌 셀 상세
    lines.append("\n**Conflict Details:**")
    for cell in ctx.get("cells", []):
        if cell["conflict_type"] == "direct":
            lines.append(
                f'  - DIRECT conflict on EC "{cell["ec"]}": '
                f'CR_A wants opposite direction to CR_B'
            )
        else:
            # 방향·상관값을 실제 데이터에서 읽어 서술 (양의 상관 경유 충돌 포함)
            dir_m = state.get("direction_matrix", {})
            d_a = dir_m.get(cell["cr_a"], {}).get(cell["ec_a"], 0)
            d_b = dir_m.get(cell["cr_b"], {}).get(cell["ec_b"], 0)
            roof_m = state.get("roof_matrix", {})
            cor = roof_m.get(cell["ec_a"], {}).get(cell["ec_b"])
            if cor is None:
                cor = roof_m.get(cell["ec_b"], {}).get(cell["ec_a"], 0)
            arrow_a = "UP" if d_a > 0 else "DOWN"
            arrow_b = "UP" if d_b > 0 else "DOWN"
            lines.append(
                f'  - ROOF conflict: CR_A wants EC "{cell["ec_a"]}" {arrow_a} '
                f'but CR_B wants EC "{cell["ec_b"]}" {arrow_b}, and these ECs '
                f'are strongly coupled (correlation {cor:+.1f}) — moving one '
                f'drags the other against the opposing CR'
            )

    # Importance Matrix 정보
    lines.append("\n**Importance Matrix (relevant CRs × ECs):**")
    imp = state.get("importance_matrix", {})
    for cr in ctx.get("cr_a_list", []) + ctx.get("cr_b_list", []) + ctx.get("cr_common_list", []):
        if cr in imp:
            nonzero = {ec: v for ec, v in imp[cr].items() if v != 0}
            if nonzero:
                lines.append(f'  "{cr[:60]}...":')
                for ec, v in nonzero.items():
                    lines.append(f"    {ec}: {v}")

    # Direction Matrix 정보
    lines.append("\n**Direction Matrix (relevant CRs × ECs):**")
    dir_m = state.get("direction_matrix", {})
    for cr in ctx.get("cr_a_list", []) + ctx.get("cr_b_list", []) + ctx.get("cr_common_list", []):
        if cr in dir_m:
            nonzero = {ec: v for ec, v in dir_m[cr].items() if v != 0}
            if nonzero:
                lines.append(f'  "{cr[:60]}...":')
                for ec, v in nonzero.items():
                    sign = "↑" if v > 0 else "↓"
                    lines.append(f"    {ec}: {sign}")

    return "\n".join(lines)


def _run_advocate(
    state: QFDState,
    segment: str,
) -> dict:
    """Segment Advocate 노드 공통 로직 (2-Phase CoT: Reasoning → JSON)"""
    seg_label = "pro" if segment == settings.SEGMENT_A else "cas"
    conflict_ctx = _build_conflict_context(state)

    # 이전 라운드 메시지 수집
    prev_messages = state.get("messages", [])
    orchestrator_msgs = [m for m in prev_messages if m["agent"] == "orchestrator"]
    last_orchestrator = orchestrator_msgs[-1] if orchestrator_msgs else None

    if state["iteration"] == 0:
        round_ctx = "This is the FIRST round. Present your opening position."
    else:
        orc_summary = json.dumps(
            last_orchestrator["content"], indent=2, ensure_ascii=False
        ) if last_orchestrator else "No previous summary."
        opponent = "cas_advocate" if segment == settings.SEGMENT_A else "pro_advocate"
        opponent_msgs = [m for m in prev_messages if m["agent"] == opponent]
        last_opponent = json.dumps(
            opponent_msgs[-1]["content"], indent=2, ensure_ascii=False
        ) if opponent_msgs else "No opponent statement yet."
        round_ctx = f"""### Previous Round Summary (from Orchestrator):
{orc_summary}

### Opponent's Last Statement:
{last_opponent}

Respond to the Orchestrator's summary and rebut/accept the opponent's arguments."""

    user_content = f"""{conflict_ctx}

{round_ctx}

Use the review search tool to find evidence supporting your segment's position.
Search format: '{segment}|<specific topic>' — be specific, not generic.
"""

    # ── Phase 1: Reasoning (도구 호출 + 자유 텍스트 분석) ──
    reasoning_system = SEGMENT_ADVOCATE_REASONING_SYSTEM.format(purpose=segment)
    print(f"    [{seg_label}_adv] Phase 1: Reasoning...")

    llm_with_tools = _chat_llm.bind_tools(ADVOCATE_TOOLS)
    messages = [
        {"role": "system", "content": reasoning_system},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4

    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [{seg_label}_adv] {tool_name}({list(tool_args.values())[0][:60]}...)")

            tool_map = {t.name: t for t in ADVOCATE_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [{seg_label}_adv] 도구 {tool_call_count}회 호출 완료")

    reasoning_text = response.content

    # ── Phase 2: JSON Structuring (reasoning 기반) ──
    print(f"    [{seg_label}_adv] Phase 2: JSON structuring...")
    json_system = SEGMENT_ADVOCATE_JSON_SYSTEM.format(purpose=segment)
    json_msg = f"""Convert the following reasoning analysis into a structured JSON argument.

### Reasoning Analysis:
{reasoning_text}

### Conflict Context (for reference):
{conflict_ctx}

### Rules:
- Follow the reasoning analysis faithfully.
- Every claim MUST have specific evidence cited in the reasoning.
- Do not add arguments not present in the reasoning."""

    result = _llm_json(json_system, json_msg)
    return result


def pro_advocate_node(state: QFDState) -> dict:
    """Professional 세그먼트 Advocate: 데이터 기반 논증 (2-Phase CoT)"""
    result = _run_advocate(state, settings.SEGMENT_A)

    return {
        "messages": [
            {
                "agent": "pro_advocate",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }


def cas_advocate_node(state: QFDState) -> dict:
    """Casual 세그먼트 Advocate: 데이터 기반 논증 (2-Phase CoT)"""
    result = _run_advocate(state, settings.SEGMENT_B)

    return {
        "messages": [
            {
                "agent": "cas_advocate",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
        "iteration": state["iteration"] + 1,
    }


def _build_numbered_cr_list(state: QFDState) -> str:
    """충돌 그룹의 모든 CR을 번호 매긴 목록으로 생성 (Orchestrator용)"""
    ctx = state.get("current_conflict", {})
    lines = []
    idx = 1
    for cr in ctx.get("cr_a_list", []):
        lines.append(f"  CR-{idx}: [{settings.SEGMENT_A}] \"{cr}\"")
        idx += 1
    for cr in ctx.get("cr_b_list", []):
        lines.append(f"  CR-{idx}: [{settings.SEGMENT_B}] \"{cr}\"")
        idx += 1
    for cr in ctx.get("cr_common_list", []):
        lines.append(f"  CR-{idx}: [Common] \"{cr}\"")
        idx += 1
    return "\n".join(lines)


def orchestrator_node(state: QFDState) -> dict:
    """토론 라운드 관리 — 양측 발언 요약, 수렴 상태 판정, 종료 판단

    그래프 순서가 pro → cas → orchestrator 이므로 이 노드는 항상
    방금 끝난 라운드(state["iteration"])의 발언을 요약한다.
    종료 판정(2라운드 연속 무변화 / 최대 라운드)은 코드가 강제한다.
    """
    conflict_ctx = _build_conflict_context(state)
    numbered_crs = _build_numbered_cr_list(state)
    _ctx = state.get("current_conflict", {})
    total_crs = (len(_ctx.get("cr_a_list", []))
                 + len(_ctx.get("cr_b_list", []))
                 + len(_ctx.get("cr_common_list", [])))
    cur_round = state["iteration"]

    # 현재까지의 토론 내용 수집
    prev_messages = state.get("messages", [])
    pro_msgs = [m for m in prev_messages if m["agent"] == "pro_advocate"]
    cas_msgs = [m for m in prev_messages if m["agent"] == "cas_advocate"]
    last_pro = json.dumps(
        pro_msgs[-1]["content"], indent=2, ensure_ascii=False
    ) if pro_msgs else "No statement."
    last_cas = json.dumps(
        cas_msgs[-1]["content"], indent=2, ensure_ascii=False
    ) if cas_msgs else "No statement."

    # 이전 라운드 Orchestrator 판정 기록 (실제 라운드만 — 준비 메시지 없음)
    orc_history = [m for m in prev_messages if m["agent"] == "orchestrator"]
    prev_states = [
        m["content"].get("convergence_state", "unknown")
        for m in orc_history if isinstance(m.get("content"), dict)
    ]

    # 직전까지의 연속 교착 라운드 수
    prev_stable = 0
    for s in reversed(prev_states):
        if s in ("stable", "deadlocked"):
            prev_stable += 1
        else:
            break

    user_content = f"""{conflict_ctx}

### ALL CRs in This Conflict Group ({total_crs} total — EVERY CR must be covered):
{numbered_crs}

### Round {cur_round} Statements:

**{settings.SEGMENT_A} Advocate:**
{last_pro}

**{settings.SEGMENT_B} Advocate:**
{last_cas}

### Previous rounds' convergence states: {prev_states}
### Consecutive stable/deadlocked rounds BEFORE this round: {prev_stable}

### Current round: {cur_round} / MAX {settings.MAX_DEBATE_ROUNDS}

### Instructions:
1. Summarize both sides fairly and classify this round's convergence_state.
2. Set should_escalate to true if:
   - Round >= {settings.MAX_DEBATE_ROUNDS}
   - This round is stable/deadlocked AND the previous round was too
     (2 consecutive rounds without position change)
   - Full convergence detected
3. If should_escalate is true, write a comprehensive final_debate_summary.
   The cr_positions array MUST contain exactly {total_crs} entries — one for
   each CR listed above. Cross-check your output against the numbered list.
   Otherwise set final_debate_summary to null."""

    llm_with_tools = _chat_llm.bind_tools(ADVOCATE_TOOLS)
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Orchestrator도 필요 시 리뷰 검색 가능
    tool_call_count = 0
    max_tool_calls = 3
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_map = {t.name: t for t in ADVOCATE_TOOLS}
            tool_name = tc["name"]
            print(f"    🔧 [orch] {tool_name}({list(tc['args'].values())[0][:60]}...)")
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(ORCHESTRATOR_SYSTEM, user_content)

    should_escalate = result.get("should_escalate", False)

    # 이번 라운드 판정을 포함한 연속 교착 라운드 수
    cur_state = result.get("convergence_state", "unknown")
    consecutive_stable = (
        prev_stable + 1 if cur_state in ("stable", "deadlocked") else 0
    )

    # ── 종료 판정 코드 강제 (매뉴얼 2-2-3) — LLM에게 위임하지 않음 ──
    if not should_escalate:
        if consecutive_stable >= 2:
            should_escalate = True
            result["should_escalate"] = True
            result.setdefault(
                "escalation_reason",
                f"code-enforced: {consecutive_stable} consecutive "
                f"stable/deadlocked rounds",
            )
            print("  [Orchestrator] 코드 강제 종료 — 2라운드 연속 무변화")
        elif cur_round >= settings.MAX_DEBATE_ROUNDS:
            should_escalate = True
            result["should_escalate"] = True
            result.setdefault(
                "escalation_reason", "code-enforced: MAX_DEBATE_ROUNDS reached"
            )
            print("  [Orchestrator] 코드 강제 종료 — 최대 라운드 도달")

    update: dict = {
        "messages": [
            {
                "agent": "orchestrator",
                "round": cur_round,
                "content": result,
            }
        ],
    }

    if should_escalate:
        update["consensus"] = True
        # 요약이 null/누락이면 마지막 라운드 요약으로 대체 (Judge 입력 보장)
        summary = (
            result.get("final_debate_summary")
            or result.get("round_summary", "")
        )
        if isinstance(summary, dict):
            summary = json.dumps(summary, ensure_ascii=False, indent=2)
        update["debate_summary"] = summary
        print(f"  [Orchestrator] 토론 종료 → Judge Phase 이관")
        print(f"    사유: {result.get('escalation_reason', 'N/A')}")
    else:
        print(f"  [Orchestrator] Round {cur_round} 완료 — "
              f"상태: {cur_state} (연속 교착 {consecutive_stable})")

    return update


def resolution_check(state: QFDState) -> str:
    """토론 종료 여부: 'continue' 또는 'end' 반환"""
    if state.get("consensus", False):
        return "end"
    if state["iteration"] >= settings.MAX_DEBATE_ROUNDS:
        return "end"
    return "continue"


def build_debate_graph():
    """Stage 2 Phase A: Advocate Debate 그래프
    pro_advocate → cas_advocate → orchestrator → resolution_check
         ↑ (continue) ←──────────────────────────────┘
                                                      ↓ (end) → END
    Orchestrator가 라운드 마지막에 요약·종료 판정을 하므로
    마지막 라운드 발언까지 요약에 포함되고, 종료 후 추가 발언이 없다.
    """
    graph = StateGraph(QFDState)

    graph.add_node("pro_advocate", pro_advocate_node)
    graph.add_node("cas_advocate", cas_advocate_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.set_entry_point("pro_advocate")
    graph.add_edge("pro_advocate", "cas_advocate")
    graph.add_edge("cas_advocate", "orchestrator")
    graph.add_conditional_edges("orchestrator", resolution_check, {
        "continue": "pro_advocate",
        "end": END,
    })

    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Judge Phase (Market Expert ↔ Engineering Expert)
# ═════════════════════════════════════════════════════════════════════════════

MIN_JUDGE_ROUNDS = 2  # Judge 토론 최소 라운드
VALID_STRATEGIES = {"Compromise", "Selection", "Separation"}


def _conflict_cr_ids(state: QFDState) -> dict[str, str]:
    """충돌 그룹 CR의 ID → 정본 매핑 (_build_numbered_cr_list와 같은 순서)"""
    ctx = state.get("current_conflict", {})
    ordered = (
        ctx.get("cr_a_list", [])
        + ctx.get("cr_b_list", [])
        + ctx.get("cr_common_list", [])
    )
    return _make_id_map(ordered, "CR")


def _judge_ec_names(state: QFDState) -> list[str]:
    """Judge가 조정 대상으로 삼는 EC 정본 목록 (ec_list 우선, 없으면 행렬에서 수집)"""
    ecs = [ec["ec"] for ec in state.get("ec_list", [])]
    if ecs:
        return ecs
    return sorted({
        ec for row in state.get("importance_matrix", {}).values() for ec in row
    })


def _conflict_intensity_profile(state: QFDState) -> tuple[str, dict]:
    """그룹의 대립 강도 프로필 — 시스템이 사실만 계산해 Judge에게 제공.

    판단 재료 제공이 목적이며 판정을 규제하지 않는다.

    Returns:
        (프롬프트용 텍스트, {"hard_cells", "has_both_segments", ...})
    """
    ctx = state.get("current_conflict", {})
    imp = state.get("importance_matrix", {})
    pri: dict[str, float] = {}
    for lst in (state.get("cr_list_a", []), state.get("cr_list_b", []),
                state.get("cr_list_common", [])):
        for c in lst:
            pri[c["cr"]] = c.get("priority", 0)

    n_pro = len(ctx.get("cr_a_list", []))
    n_cas = len(ctx.get("cr_b_list", []))
    n_common = len(ctx.get("cr_common_list", []))

    hard = hard_other = mid = weak = roof_n = 0
    for cell in ctx.get("cells", []):
        if cell.get("conflict_type") == "roof":
            roof_n += 1
            continue
        ec = cell.get("ec", "")
        ia = imp.get(cell["cr_a"], {}).get(ec, 0)
        ib = imp.get(cell["cr_b"], {}).get(ec, 0)
        if ia == 9 and ib == 9:
            if pri.get(cell["cr_a"], 0) >= 4 and pri.get(cell["cr_b"], 0) >= 4:
                hard += 1
            else:
                hard_other += 1
        elif 9 in (ia, ib):
            mid += 1
        else:
            weak += 1

    text = (
        "### Conflict Intensity Profile (computed by the system):\n"
        f"- Segments in this group: Pro {n_pro} CRs, Cas {n_cas} CRs, "
        f"Common {n_common} CRs\n"
        f"- Opposing cells by strength: 9-vs-9 (both priority>=4): {hard} | "
        f"9-vs-9 (other): {hard_other} | 9-vs-3: {mid} | 3-vs-3: {weak}\n"
        f"- Roof (indirect) cells: {roof_n}"
    )
    meta = {
        "hard_cells": hard,
        "has_both_segments": n_pro > 0 and n_cas > 0,
        "hard_other": hard_other, "mid": mid, "weak": weak, "roof": roof_n,
    }
    return text, meta


def _build_judge_context(state: QFDState) -> str:
    """Judge Phase에 전달할 전체 컨텍스트"""
    lines = ["### Debate Summary (from Orchestrator)"]
    lines.append(state.get("debate_summary", "No summary available."))

    lines.append("\n### Conflict Group Details")
    lines.append(_build_conflict_context(state))

    # Roof Matrix 정보 (기술적 상관관계)
    roof = state.get("roof_matrix", {})
    if roof:
        lines.append("\n### Roof Matrix (EC correlations):")
        for ec_a, correlations in roof.items():
            for ec_b, val in correlations.items():
                lines.append(f"  {ec_a} ↔ {ec_b}: {val}")

    # 대립 강도 프로필 — 시스템이 계산한 판단 재료 (판정은 Judge의 몫)
    profile_text, _ = _conflict_intensity_profile(state)
    lines.append("\n" + profile_text)

    # 조정 키로 쓸 CR/EC ID 목록 — LLM은 ID로만 답하고 코드가 정본으로 치환
    cr_ids = _conflict_cr_ids(state)
    lines.append("\n### CR IDs (use as final_adjustments keys):")
    for cid, cr in cr_ids.items():
        lines.append(f'{cid}: "{cr}"')

    ec_names = _judge_ec_names(state)
    if ec_names:
        lines.append("\n### EC IDs (use as inner keys):")
        for i, name in enumerate(ec_names, 1):
            lines.append(f'EC-{i}: "{name}"')

    # 10/10: 조정 가능 셀 = 이 그룹에 배속된 충돌 셀 (EC 기준 그룹화와 짝)
    scope = _group_scope_cells(state)
    if scope:
        cr_id_of = {name: cid for cid, name in cr_ids.items()}
        ec_id_of = {name: f"EC-{i}" for i, name in enumerate(ec_names, 1)}
        imp = state.get("importance_matrix", {})
        lines.append("\n### Adjustable cells (this group's conflict cells — the ONLY "
                     "cells you may change; other cells are discarded):")
        def _key(t: tuple[str, str]) -> tuple[int, int]:
            c = cr_id_of.get(t[0], "CR-999"); e = ec_id_of.get(t[1], "EC-999")
            return (int(c.split("-")[1]), int(e.split("-")[1]))
        for cr, ec in sorted(scope, key=_key):
            lines.append(f"{cr_id_of.get(cr, cr[:40])} × {ec_id_of.get(ec, ec[:40])}: "
                         f"current importance {imp.get(cr, {}).get(ec, 0)}")

    return "\n".join(lines)


def _format_validator_feedback(state: QFDState, for_engineering: bool) -> str:
    """Validator 피드백을 두 종류로 나눠 전달 (발송 9 #14)

    - 오류(반드시 수정): "VALIDATION ERRORS" 헤더
    - 재고 요청(오류 아님, 성찰): 별도 헤더 — 강제가 아니라는 점을 명시
    """
    validator_msgs = [m for m in state["messages"] if m["agent"] == "validator"]
    if not validator_msgs:
        return ""
    last_val = validator_msgs[-1]["content"]
    if last_val.get("action") != "retry_requested":
        return ""
    errors = last_val.get("errors", [])
    recon = [e for e in errors if e.startswith("[RECONSIDERATION REQUEST")]
    hard = [e for e in errors if e not in recon]
    parts: list[str] = []
    if hard:
        tail = (
            "You MUST fix ALL validation errors above. Your response will be "
            "rejected again if these issues remain unresolved."
            if for_engineering else
            "You MUST address ALL validation errors above in your revised proposal."
        )
        parts.append(
            "### ⚠️ VALIDATION ERRORS (from previous round — MUST be fixed):\n"
            + "\n".join(f"  - {e}" for e in hard) + "\n\n" + tail
        )
    if recon:
        parts.append(
            "### Reconsideration request (NOT an error — reflect, then decide):\n"
            + "\n".join(f"  - {e}" for e in recon)
            + "\n\nYou may keep your decision if you rebut the profile with "
            "specific evidence; it will then be accepted."
        )
    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


def _count_cells(adj: dict) -> int:
    return sum(len(r) for r in adj.values() if isinstance(r, dict))


def _group_scope_cells(state: QFDState) -> set[tuple[str, str]]:
    """이 그룹에 배속된 충돌 셀 (CR, EC) 집합 — Judge가 조정할 수 있는 유일한 범위 (10/10).

    direct 셀은 (cr_a, ec)·(cr_b, ec), roof 셀은 (cr_a, ec_a)·(cr_b, ec_b)."""
    cells: set[tuple[str, str]] = set()
    for c in state.get("current_conflict", {}).get("cells", []):
        if c.get("conflict_type") == "direct":
            cells.add((c["cr_a"], c["ec"]))
            cells.add((c["cr_b"], c["ec"]))
        else:
            cells.add((c["cr_a"], c["ec_a"]))
            cells.add((c["cr_b"], c["ec_b"]))
    return cells


def _resolve_and_guard_adjustments(
    state: QFDState, raw: dict
) -> tuple[dict, list[str]]:
    """LLM의 final_adjustments(ID 키) → 정본 이름 치환 → 셀 가드 → 그룹 범위 가드.

    정상 경로(Eng finalized)와 라운드 소진 경로가 반드시 같은 처리를 거친다
    (발송 9 #1·#2 — 예비 경로가 ID 치환을 건너뛰던 결함 제거).

    Returns:
        (가드를 통과한 조정, 그룹 밖이라 폐기된 셀 설명 목록)"""
    cr_ids = _conflict_cr_ids(state)
    ec_ids = _make_id_map(_judge_ec_names(state), "EC")
    adj = _resolve_matrix_ids(raw or {}, cr_ids, ec_ids, "Judge")
    original = state.get("importance_matrix", {})
    # 10/10 (4-2): 0/3/9 밖의 값은 여기서 보정하지 않는다. 검증기가 [INVALID VALUE]로
    # 반려해 Judge가 다시 내게 하고, MAX 라운드 폴백에서만 기록을 남기며 보정한다.
    guarded = _guard_existing_cells_only(adj, original)
    # 10/10: 이 그룹에 배속된 충돌 셀만 조정 가능 — 다른 그룹의 셀은 폐기
    # (EC 기준 그룹화와 짝: 셀의 소속 그룹이 하나로 정해지므로 그룹 간 덮어쓰기가 사라진다)
    scope = _group_scope_cells(state)
    if not scope:
        return guarded, []
    kept: dict = {}
    violations: list[str] = []
    for cr, row in guarded.items():
        for ec, val in row.items():
            if (cr, ec) in scope:
                kept.setdefault(cr, {})[ec] = val
            else:
                violations.append(f'"{cr[:40]}" × "{ec[:40]}" = {val}')
    if violations:
        print(f"  ⚠️  [Judge 가드] 그룹 밖 셀 {len(violations)}개 폐기 "
              f"— 조정은 이 그룹의 충돌 셀로 제한")
    return kept, violations


def market_expert_judge_node(state: QFDState) -> dict:
    """Market Expert (Judge): 시장/고객 관점에서 전략 제안 및 수치 조정 제안"""
    judge_ctx = _build_judge_context(state)
    validator_feedback = _format_validator_feedback(state, for_engineering=False)

    if state["iteration"] == 0:
        user_content = f"""{judge_ctx}

### Task:
Based on the debate summary and conflict data, propose:
1. A resolution strategy (evaluate in order: Compromise → Selection → Separation)
2. Specific importance matrix adjustments for the conflicting CRs

FIRST: Use your search tools to gather market evidence.
THEN: Propose your strategy and adjustments.

Set "finalized": false — the Engineering Expert will review your proposal."""
    else:
        eng_msgs = [m for m in state["messages"] if m["agent"] == "engineering_expert_judge"]
        last_eng = json.dumps(
            eng_msgs[-1]["content"], indent=2, ensure_ascii=False
        ) if eng_msgs else "No response yet."

        user_content = f"""{judge_ctx}

### Engineering Expert's Response:
{last_eng}
{validator_feedback}
Address the Engineering Expert's feedback.
- For ACCEPT items: Acknowledge and keep.
- For REJECT items: Provide market counter-evidence, or concede.

You may use search tools for additional evidence.
Set "finalized": false."""

    llm_with_tools = _chat_llm.bind_tools(JUDGE_MARKET_TOOLS)
    messages = [
        {"role": "system", "content": MARKET_EXPERT_JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [mkt_judge] {tool_name}({list(tool_args.values())[0][:60]}...)")
            tool_map = {t.name: t for t in JUDGE_MARKET_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [mkt_judge] 도구 {tool_call_count}회 호출 완료")

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(MARKET_EXPERT_JUDGE_SYSTEM, user_content)

    return {
        "iteration": state["iteration"] + 1,
        "messages": [
            {
                "agent": "market_expert_judge",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }


def engineering_expert_judge_node(state: QFDState) -> dict:
    """Engineering Expert (Judge): 기술 관점에서 전략 검토 및 최종 수치 결정"""
    judge_ctx = _build_judge_context(state)

    mkt_msgs = [m for m in state["messages"] if m["agent"] == "market_expert_judge"]
    last_mkt = json.dumps(
        mkt_msgs[-1]["content"], indent=2, ensure_ascii=False
    ) if mkt_msgs else "No proposal yet."

    validator_feedback = _format_validator_feedback(state, for_engineering=True)

    user_content = f"""{judge_ctx}

### Market Expert's Proposal:
{last_mkt}
{validator_feedback}
### Task:
Review the Market Expert's proposed strategy and importance adjustments.
For EACH point:
- ACCEPT: if technically sound.
- REJECT: if it creates engineering inconsistencies, violates Roof Matrix
  constraints, or lacks technical evidence. Provide counter-evidence.

FIRST: Use your search tools to verify technical feasibility.
THEN: Decide.

Set "finalized": true when you have addressed ALL feedback points AND the
adjusted values are technically consistent. Remember: you have final authority."""

    llm_with_tools = _chat_llm.bind_tools(JUDGE_ENGINEERING_TOOLS)
    messages = [
        {"role": "system", "content": ENGINEERING_EXPERT_JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [eng_judge] {tool_name}({list(tool_args.values())[0][:60]}...)")
            tool_map = {t.name: t for t in JUDGE_ENGINEERING_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [eng_judge] 도구 {tool_call_count}회 호출 완료")

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(ENGINEERING_EXPERT_JUDGE_SYSTEM, user_content)

    update: dict = {
        "messages": [
            {
                "agent": "engineering_expert_judge",
                "round": state["iteration"],
                "content": result,
            }
        ],
    }

    # finalized 여부만 기록 — 실제 consensus 판정은 judge_validator_node에서 수행
    finalized = result.get("finalized", False)
    n_raw = _count_cells(result.get("final_adjustments", {}) or {})
    # 라운드별 전략 이력 — 반려 뒤 전략을 바꿔 회피했는지 로그로 확인 가능
    print(f"  [Judge/Eng] Round {state['iteration']} 전략: "
          f"{result.get('agreed_strategy', '?')} · finalized={finalized} · "
          f"제출 조정 셀 {n_raw}개")
    if finalized:
        strategy = result.get("agreed_strategy", "Compromise")
        update["judge_strategy"] = strategy
        update["selected_segment"] = result.get("selected_segment")

        # 전략과 무관하게 단일 final_adjustments 사용 — 변경 셀만 담기며,
        # 미변경 셀은 stage2의 오버레이가 원본을 유지한다.
        # Separation의 모델 분할은 Stage 3가 CR의 세그먼트 소속으로 수행한다.
        update["adjusted_importance"], update["scope_violations"] = (
            _resolve_and_guard_adjustments(
                state, result.get("final_adjustments", {})
            )
        )
        print(f"  [Judge/Eng] 최종 결정 제출: {strategy} → Validator 검증 대기")
    else:
        update["consensus"] = False
        print(f"  [Judge/Eng] Round {state['iteration']} — 추가 토론 필요")

    return update


def _guard_existing_cells_only(adj: dict, original: dict) -> dict:
    """Judge 조정은 기존 관계(원본 importance > 0)의 값 조정만 허용.

    원본이 0인 (CR, EC) 셀에 값을 부여하는 것(신규 관계 생성)은 설계에 없는
    오류이므로 폐기하고 경고한다. 관계의 존재 여부는 Stage 1 토론의 권한."""
    guarded: dict = {}
    dropped = 0
    unchanged = 0
    for cr, row in adj.items():
        kept: dict = {}
        for ec, val in row.items():
            if val != 0 and original.get(cr, {}).get(ec, 0) == 0:
                dropped += 1
                continue
            # 원본과 같은 값은 조정이 아님 — 문서 2-3-3 "변경 셀만" (10/1: Judge가
            # "기존 값 재확인" 셀을 조정으로 제출해 0건 반려 규칙이 우회되던 문제)
            if val == original.get(cr, {}).get(ec, 0):
                unchanged += 1
                continue
            kept[ec] = val
        if kept:
            guarded[cr] = kept
    if unchanged:
        print(f"  ℹ️  [Judge 가드] 원본과 같은 값 {unchanged}개는 조정으로 세지 않음")
    if dropped:
        print(f"  ⚠️  [Judge 가드] 원본 관계가 0인 셀에 대한 신규 값 {dropped}개 폐기 "
              f"— Judge는 기존 관계의 조정만 가능")
    return guarded


EMPTY_ADJ_PREFIX = "[EMPTY ADJUSTMENTS]"
SCOPE_PREFIX = "[OUT OF GROUP SCOPE]"
INVALID_VALUE_PREFIX = "[INVALID VALUE]"


def _snap_adjustments_directional(adjusted: dict, original: dict) -> tuple[dict, int]:
    """MAX 라운드 폴백: 0/3/9 밖의 조정값을 '변경 방향의 가장 가까운 유효값'으로 보정.
    (9→7 은 9→3, 3→1 은 3→0, 0→2 는 0→3). 방향이 없으면 원값 유지."""
    valid = sorted(VALID_IMPORTANCE_VALUES)
    out: dict = {}; n = 0
    for cr, row in adjusted.items():
        new_row: dict = {}
        for ec, v in row.items():
            if v in VALID_IMPORTANCE_VALUES:
                new_row[ec] = v; continue
            o = original.get(cr, {}).get(ec, 0)
            if v < o:
                cands = [x for x in valid if x < o]; nv = max(cands) if cands else o
            elif v > o:
                cands = [x for x in valid if x > o]; nv = min(cands) if cands else o
            else:
                nv = o
            new_row[ec] = nv; n += 1
        out[cr] = new_row
    return out, n


def _validate_judge_result(
    strategy: str,
    selected_segment: str | None,
    adjusted: dict,
) -> list[str]:
    """Engineering Expert Judge의 최종 결과를 검증하여 오류 목록 반환

    발송 8: final_adjustments는 "변경 셀만" 제출 방식 — 미변경 CR은 원본이
    유지되므로 CR 누락 검사는 폐지.
    발송 9 #1: Compromise/Selection인데 조정 셀이 0개이면 오류 — 매 라운드 반려."""
    errors: list[str] = []

    # ── 공통 검증 ──
    if strategy not in VALID_STRATEGIES:
        errors.append(
            f'agreed_strategy is invalid: "{strategy}". '
            f"Allowed values: {VALID_STRATEGIES}"
        )
        return errors

    # ── Strategy별 검증 ──
    if strategy == "Selection":
        valid_segments = {settings.SEGMENT_A, settings.SEGMENT_B}
        if selected_segment not in valid_segments:
            errors.append(
                f"Strategy is Selection but selected_segment is missing. "
                f"You MUST specify one of {valid_segments}."
            )

    bad = [(cr, ec, v) for cr, row in adjusted.items() if isinstance(row, dict)
           for ec, v in row.items() if v not in VALID_IMPORTANCE_VALUES]
    if bad:
        errors.append(
            f"{INVALID_VALUE_PREFIX} importance values must be 0, 3 or 9 — "
            "the HoQ scale has no intermediate steps. Invalid cells: "
            + "; ".join(f'"{cr[:40]}" × "{ec[:40]}" = {v}' for cr, ec, v in bad[:8])
            + ". Re-express each as 0 (remove influence), 3 (secondary) or 9 (primary)."
        )

    if strategy in ("Compromise", "Selection") and _count_cells(adjusted) == 0:
        errors.append(
            f"{EMPTY_ADJ_PREFIX} final_adjustments is empty, but the strategy "
            f"is {strategy}. Under Compromise/Selection the resolution is "
            "expressed ONLY through changed importance cells, so an empty "
            "submission is rejected. Translate the debate summary's "
            "concessions and agreements into at least one changed cell "
            "(this group's adjustable cells only, values 0|3|9), each with a rationale. "
            "Do NOT switch strategy to avoid this — a strategy change must be "
            "justified on its own merits."
        )

    return errors


def _fallback_defaults(strategy: str, selected_segment: str | None) -> dict:
    """MAX_JUDGE_ROUNDS 도달 시 형식 요건의 기본값 보장 (Selection 세그먼트)"""
    update: dict = {}
    if strategy == "Selection" and not selected_segment:
        update["selected_segment"] = settings.SEGMENT_A
        print(
            f"  [Validator/Fallback] selected_segment 미지정 → "
            f"기본값 '{settings.SEGMENT_A}' 적용"
        )
    return update


def _translate_concessions_to_adjustments(
    state: QFDState,
    strategy: str,
    selected_segment: str | None,
) -> tuple[dict, dict]:
    """마지막 라운드에도 조정이 비었을 때의 번역 전용 호출 1회 (발송 9 #1)

    전략은 확정된 것으로 두고, 토론 요약의 양보·합의를 셀 조정으로 옮기기만
    요청한다. 도구 없음. 결과는 정상 경로와 같은 ID 치환·셀 가드를 거친다.

    Returns:
        (adjusted_importance, raw_llm_result)
    """
    judge_ctx = _build_judge_context(state)
    mkt_msgs = [m for m in state["messages"] if m["agent"] == "market_expert_judge"]
    eng_msgs = [m for m in state["messages"] if m["agent"] == "engineering_expert_judge"]
    mkt_reason = mkt_msgs[-1]["content"].get("market_reasoning", "") if mkt_msgs else ""
    eng_reason = eng_msgs[-1]["content"].get("engineering_reasoning", "") if eng_msgs else ""
    user = f"""{judge_ctx}

### Decided strategy: {strategy}
### Selected segment (Selection only): {selected_segment}

### Market Judge's last reasoning:
{mkt_reason}

### Engineering Judge's last reasoning:
{eng_reason}

### Task:
The strategy above is final. Translate the debate summary's concessions,
agreed middle grounds, and boundaries into importance adjustments on the
existing cells listed in the context. Output JSON only."""
    result = _llm_json(JUDGE_ADJUSTMENT_TRANSLATION_SYSTEM, user)
    adjusted, _ = _resolve_and_guard_adjustments(state, result.get("final_adjustments", {}))
    return adjusted, result


def judge_validator_node(state: QFDState) -> dict:
    """Judge 결과 검증 노드 — Strategy별 필수 조건 확인 (발송 9 #1)

    - finalized & 검증 통과 → consensus=True
    - 검증 실패 + 라운드 여유 → 재시도 요청 (조정 0건은 매 라운드 반려)
    - MAX 도달 & finalized=false → 마지막 제안을 정상 경로(ID 치환·셀 가드)로 채택 후 검증
    - MAX 도달 & 조정 0건 → 번역 전용 호출 1회 → 그래도 0건이면 judge_failed
      (Stage 2 중단 — 0건 조정이 결과에 조용히 들어가는 일은 없다)
    """
    eng_msgs = [m for m in state["messages"] if m["agent"] == "engineering_expert_judge"]
    if not eng_msgs:
        return {"consensus": False}

    last_eng = eng_msgs[-1]["content"]
    finalized = last_eng.get("finalized", False)
    at_max = state["iteration"] >= settings.MAX_JUDGE_ROUNDS
    update: dict = {}

    if not finalized:
        if not at_max:
            return {"consensus": False}
        # 라운드 소진 — 마지막 제안을 정상 경로와 동일하게 처리 (구 예비 경로 대체)
        print(f"  [Validator] MAX_JUDGE_ROUNDS 도달, finalized=false → "
              f"마지막 제안을 정규 경로(ID 치환·셀 가드)로 채택")
        strategy = (last_eng.get("agreed_strategy")
                    or state.get("judge_strategy") or "Compromise")
        selected_segment = (last_eng.get("selected_segment")
                            or state.get("selected_segment"))
        adjusted, scope_violations = _resolve_and_guard_adjustments(
            state, last_eng.get("final_adjustments", {})
        )
        update.update({
            "judge_strategy": strategy,
            "selected_segment": selected_segment,
            "adjusted_importance": adjusted,
            "scope_violations": scope_violations,
        })
    else:
        # ── 최소 라운드 검증 ──
        if state["iteration"] < MIN_JUDGE_ROUNDS:
            print(f"  [Validator] 최소 라운드 미달 ({state['iteration']}/{MIN_JUDGE_ROUNDS})")
            return {"consensus": False}
        strategy = state.get("judge_strategy", "")
        selected_segment = state.get("selected_segment")
        adjusted = state.get("adjusted_importance") or {}
        scope_violations = state.get("scope_violations") or []

    # ── Strategy-aware Validation ──
    errors = _validate_judge_result(strategy, selected_segment, adjusted)
    # 10/10: 그룹 밖 셀을 조정하려 했으면 반려 (라운드 여유가 있을 때) — Judge가 범위 안에서 다시 내게 한다
    if scope_violations and not at_max:
        errors.append(
            f"{SCOPE_PREFIX} these cells belong to another conflict group and were "
            "discarded: " + "; ".join(scope_violations[:8]) + ". Adjust ONLY the cells "
            "listed under 'Adjustable cells' in the context."
        )

    # ── Separation 재고 요청 (성찰 — 오류 아님, 1회만) ──
    # 한 세그먼트만 있는 그룹에서 Separation을 골랐을 때만 재고를 요청한다.
    # 9-vs-9 개수는 프롬프트와 같이 참고 정보일 뿐 판정 조건이 아니다 (10/10).
    # 재고 후에도 Separation을 유지하면 그대로 수용한다 — 최종 결정은 Judge의 것.
    if (strategy == "Separation"
            and not state.get("separation_reconsidered", False)
            and not at_max):
        _, prof = _conflict_intensity_profile(state)
        reasons = []
        if not prof["has_both_segments"]:
            reasons.append("only one segment is present in this group, so "
                           "splitting by segment resolves nothing")
        if reasons:
            msg = (
                "[RECONSIDERATION REQUEST - not an error] You chose Separation, "
                "but " + " and ".join(reasons) + ". Reconsider whether Compromise "
                "or Selection resolves this group. If you still judge Separation "
                "to be right, restate it WITH the physical reason (the contested "
                "ECs need different levels per segment and the Roof Matrix couples "
                "them) - your decision will then be accepted."
            )
            print(f"  [Validator] Separation 재고 요청 — 프로필: "
                  f"강대강 {prof['hard_cells']}개, "
                  f"양세그={prof['has_both_segments']}")
            return {
                "consensus": False,
                "separation_reconsidered": True,
                "messages": [{
                    "agent": "validator",
                    "round": state["iteration"],
                    "content": {
                        "validation_passed": False,
                        "errors": errors + [msg],
                        "action": "retry_requested",
                    },
                }],
            }

    if not errors:
        if strategy == "Separation" and state.get("separation_reconsidered"):
            print(f"  [Validator] 재고 후 Separation 유지 — 수용")
        print(f"  [Validator] ✅ 검증 통과 — {strategy} 확정 "
              f"(조정 셀 {_count_cells(adjusted)}개)")
        update["consensus"] = True
        return update

    # 검증 실패
    for err in errors:
        print(f"  [Validator] ❌ {err[:140]}")

    if not at_max:
        # 라운드 여유 있음 → 재시도 요청 (조정 0건 포함, 매 라운드)
        print(f"  [Validator] 검증 실패 → 재시도 요청 "
              f"(Round {state['iteration']}/{settings.MAX_JUDGE_ROUNDS})")
        return {
            "consensus": False,
            "messages": [{
                "agent": "validator",
                "round": state["iteration"],
                "content": {
                    "validation_passed": False,
                    "errors": errors,
                    "action": "retry_requested",
                },
            }],
        }

    # ── MAX 도달 + 오류 ──
    update.update(_fallback_defaults(strategy, selected_segment))
    if any(e.startswith(INVALID_VALUE_PREFIX) for e in errors):
        snapped, n_snap = _snap_adjustments_directional(
            adjusted, state.get("importance_matrix", {})
        )
        print(f"  [Validator/Fallback] 0/3/9 밖 조정값 {n_snap}개 → 변경 방향의 유효값으로 보정")
        adjusted = snapped
        update["adjusted_importance"] = snapped
        errors = [e for e in errors if not e.startswith(INVALID_VALUE_PREFIX)]
        if not errors:
            print(f"  [Validator] ✅ 보정 후 검증 통과 — {strategy} 확정 (조정 셀 {_count_cells(adjusted)}개)")
            update["consensus"] = True
            update["messages"] = [{
                "agent": "validator", "round": state["iteration"],
                "content": {"validation_passed": True, "errors": [],
                            "action": "fallback_applied", "snapped_values": n_snap},
            }]
            return update
    if not any(e.startswith(EMPTY_ADJ_PREFIX) for e in errors):
        # 형식 오류만 남음 → 기본값 보완 후 수용 (기존 동작)
        print(f"  [Validator] MAX_JUDGE_ROUNDS 도달 → fallback 보완 적용")
        update["consensus"] = True
        update["messages"] = [{
            "agent": "validator",
            "round": state["iteration"],
            "content": {
                "validation_passed": False,
                "errors": errors,
                "action": "fallback_applied",
            },
        }]
        return update

    # 조정 0건이 끝까지 남음 → 번역 전용 호출 1회
    print(f"  [Validator] MAX 도달 — 조정 0건 → 번역 전용 호출 1회 ({strategy})")
    translated, raw = _translate_concessions_to_adjustments(
        state, strategy, update.get("selected_segment", selected_segment)
    )
    n = _count_cells(translated)
    msgs = [{
        "agent": "judge_translation",
        "round": state["iteration"],
        "content": raw,
    }]
    if n > 0:
        print(f"  [Validator] 번역 호출 성공 — 조정 셀 {n}개 채택")
        update["adjusted_importance"] = translated
        update["consensus"] = True
        msgs.append({
            "agent": "validator",
            "round": state["iteration"],
            "content": {
                "validation_passed": True,
                "errors": errors,
                "action": "translation_applied",
                "cells": n,
            },
        })
        update["messages"] = msgs
        return update

    # 10/13: 끝까지 0건이면 결과에 넣지 않고 중단 (발송 9 #1 원래 동작 복원).
    # "조정 없는 Compromise/Selection"은 산출물에 남지 않는다 — run_conflict_resolution이 중단 처리.
    reason = (
        f"Strategy {strategy}, but zero adjusted cells remained after "
        f"{settings.MAX_JUDGE_ROUNDS} rejected rounds and the translation call "
        f"(translation call reason: {raw.get('reason', 'N/A')})"
    )
    print(f"  ❌ [Validator] {reason}")
    update["judge_failed"] = True
    update["judge_failure_reason"] = reason
    update["adjusted_importance"] = {}
    update["consensus"] = True  # 그래프 종료용 — run_conflict_resolution이 중단 처리
    msgs.append({
        "agent": "validator",
        "round": state["iteration"],
        "content": {
            "validation_passed": False,
            "errors": errors,
            "action": "judge_failed",
            "reason": reason,
        },
    })
    update["messages"] = msgs
    return update


def judge_validation_route(state: QFDState) -> str:
    """Validator 이후 라우팅: validated → END, retry → market_expert_judge"""
    if state.get("consensus", False):
        return "end"
    if state["iteration"] >= settings.MAX_JUDGE_ROUNDS:
        return "end"
    return "retry"


def build_judge_graph():
    """Stage 2 Phase B: Judge 토론 그래프
    market_expert_judge → engineering_expert_judge → judge_validator
         ↑ (retry) ←────────────────────────────────────┘
                                                         ↓ (end) → END
    """
    graph = StateGraph(QFDState)

    graph.add_node("market_expert_judge", market_expert_judge_node)
    graph.add_node("engineering_expert_judge", engineering_expert_judge_node)
    graph.add_node("judge_validator", judge_validator_node)

    graph.set_entry_point("market_expert_judge")
    graph.add_edge("market_expert_judge", "engineering_expert_judge")
    graph.add_edge("engineering_expert_judge", "judge_validator")
    graph.add_conditional_edges("judge_validator", judge_validation_route, {
        "retry": "market_expert_judge",
        "end": END,
    })

    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Combined Resolution Runner (Debate + Judge 순차 실행)
# ═════════════════════════════════════════════════════════════════════════════

def run_conflict_resolution(
    conflict_group: dict,
    hoq_data: dict,
    conflict_matrix: dict,
    group_index: int = 1,
) -> dict:
    """하나의 충돌 그룹에 대해 Debate → Judge 순차 실행

    Args:
        conflict_group: ConflictGroup.model_dump() — cr_a_list, cr_b_list, cells
        hoq_data: hoq.json 전체 데이터
        conflict_matrix: 전체 N×N Conflict Matrix
        group_index: 그룹 번호 (로그 파일명용)

    Returns:
        {
            "strategy": str,
            "reasoning": str,
            "debate_summary": str,
            "judge_discussion_log": list[dict],
            "adjusted_importance": dict,
        }
    """
    # ── 충돌 그룹에 관련된 CR 리스트 구성 ──
    # Specific Pro CRs
    pro_crs = hoq_data.get("specific_professional", {}).get("cr_list", [])
    cas_crs = hoq_data.get("specific_casual", {}).get("cr_list", [])

    # 전체 Importance / Direction Matrix 통합 (specific 블록)
    all_importance = {}
    all_direction = {}
    for block_key in ["specific_professional", "specific_casual", "common"]:
        block = hoq_data.get(block_key, {})
        all_importance.update(block.get("importance_matrix", {}))
        all_direction.update(block.get("direction_matrix", {}))

    # ── Phase A: Advocate Debate ──
    print(f"\n  [Debate] Advocate 토론 시작")
    print(f"    Pro CRs: {len(conflict_group['cr_a_list'])}개")
    print(f"    Cas CRs: {len(conflict_group['cr_b_list'])}개")
    if conflict_group.get("cr_common_list"):
        print(f"    Common CRs: {len(conflict_group['cr_common_list'])}개")
    print(f"    충돌 셀: {len(conflict_group['cells'])}개")

    debate_state: QFDState = {
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "cr_list_a": pro_crs,
        "cr_list_b": cas_crs,
        "cr_list_common": hoq_data.get("common", {}).get("cr_list", []),
        "current_conflict": conflict_group,
        "ec_list": hoq_data.get("ec_list", []),
        "importance_matrix": all_importance,
        "direction_matrix": all_direction,
        "roof_matrix": hoq_data.get("roof_matrix", {}),
        "conflict_matrix": conflict_matrix,
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }

    debate_graph = build_debate_graph()
    debate_result = debate_graph.invoke(debate_state)

    debate_summary = debate_result.get("debate_summary", "")
    debate_rounds = debate_result["iteration"]
    print(f"  [Debate] 토론 완료: {debate_rounds}라운드")

    # debate_summary가 비어있으면 마지막 orchestrator 메시지에서 추출
    if not debate_summary:
        orc_msgs = [
            m for m in debate_result["messages"]
            if m["agent"] == "orchestrator"
        ]
        if orc_msgs:
            last_orc = orc_msgs[-1]["content"]
            debate_summary = (
                last_orc.get("final_debate_summary")
                or last_orc.get("round_summary", "")
            )

    # ── Phase B: Judge (2-Agent) ──
    print(f"\n  [Judge] 전략 결정 토론 시작")

    judge_state: QFDState = {
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "cr_list_a": pro_crs,
        "cr_list_b": cas_crs,
        "cr_list_common": hoq_data.get("common", {}).get("cr_list", []),
        "current_conflict": conflict_group,
        "ec_list": hoq_data.get("ec_list", []),
        "importance_matrix": all_importance,
        "direction_matrix": all_direction,
        "roof_matrix": hoq_data.get("roof_matrix", {}),
        "conflict_matrix": conflict_matrix,
        "debate_summary": debate_summary,
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }

    judge_graph = build_judge_graph()
    judge_result = judge_graph.invoke(judge_state)

    judge_rounds = judge_result["iteration"]
    strategy = judge_result.get("judge_strategy", "Compromise")
    adjusted = judge_result.get("adjusted_importance", {}) or {}
    # 발송 9: 라운드 소진 시의 채택은 judge_validator_node가 정상 경로(ID 치환·
    # 셀 가드)로 수행한다 — 마지막 응답을 그대로 가져오던 예비 경로는 제거.

    selected_segment = judge_result.get("selected_segment")

    print(f"  [Judge] 완료: {judge_rounds}라운드 → 전략: {strategy}")
    if strategy == "Selection" and selected_segment:
        print(f"    선택 세그먼트: {selected_segment}")
    elif strategy == "Separation":
        print(f"    Separation → Stage 3가 CR의 세그먼트 소속으로 2모델 분할")

    # 토론 로그 저장
    save_debate_log(
        f"stage2_group{group_index}_advocate_debate.txt",
        debate_result["messages"],
        f"Stage 2 Group {group_index} — Advocate Debate "
        f"(Pro {len(conflict_group['cr_a_list'])} × Cas {len(conflict_group['cr_b_list'])}"
        f"{' + Common ' + str(len(conflict_group.get('cr_common_list', []))) if conflict_group.get('cr_common_list') else ''}, "
        f"{debate_rounds} rounds)",
    )
    save_debate_log(
        f"stage2_group{group_index}_judge_debate.txt",
        judge_result["messages"],
        f"Stage 2 Group {group_index} — Judge Phase "
        f"({judge_rounds} rounds, strategy: {strategy})",
    )

    # 10/13: 조정 0건이 끝까지 해소되지 않으면 여기서 중단 (로그는 위에서 저장됨).
    # 조정 없는 Compromise/Selection을 결과에 넣지 않는다 — 다시 실행해야 한다.
    if judge_result.get("judge_failed") or (
        strategy in ("Compromise", "Selection")
        and not any(r for r in adjusted.values() if isinstance(r, dict) and r)
    ):
        raise RuntimeError(
            f"[Stage 2] Group {group_index} Judge 실패 — "
            f"{judge_result.get('judge_failure_reason', f'{strategy} 전략인데 조정 셀 0개')}. "
            f"로그: {LOG_DIR}/stage2_group{group_index}_judge_debate.txt. "
            f"`uv run python3 main.py --only 2` 로 다시 실행하세요."
        )

    # Judge 토론 로그 수집
    judge_log = [
        m["content"] for m in judge_result["messages"]
        if m["agent"] in ("market_expert_judge", "engineering_expert_judge")
    ]

    # Engineering Expert의 reasoning 추출
    eng_msgs = [
        m for m in judge_result["messages"]
        if m["agent"] == "engineering_expert_judge"
    ]
    reasoning = ""
    if eng_msgs:
        reasoning = eng_msgs[-1]["content"].get("engineering_reasoning", "")

    result_dict = {
        "strategy": strategy,
        "reasoning": reasoning,
        "debate_summary": debate_summary,
        "judge_discussion_log": judge_log,
        "adjusted_importance": adjusted,
        "debate_rounds": debate_rounds,
        "judge_rounds": judge_rounds,
    }

    if selected_segment:
        result_dict["selected_segment"] = selected_segment

    return result_dict


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Part Deployment — PC Debate (Phase 2-A)
# EC → Part Characteristics 도출 토론
# ═════════════════════════════════════════════════════════════════════════════

MIN_PC_DEBATE_ROUNDS = 2  # PC 토론 최소 라운드
MIN_PC_ROOF_DEBATE_ROUNDS = 2  # PC Roof 토론 최소 라운드
PC_MATRIX_FILL_CHUNK_SIZE = 5  # EC 분할 단위 (EC-PC Matrix Fill)


def _compute_pc_range(state: QFDState) -> tuple[int, int]:
    """EC 수에 연동하여 PC 수 범위를 결정 (상한은 settings.MAX_PCS)"""
    n_ecs = len(state.get("ec_list", []))
    max_pcs = min(settings.MAX_PCS, max(12, round(n_ecs * 1.4)))
    min_pcs = max(8, max_pcs // 2)
    return min_pcs, max_pcs


# ── Separation 제품군 특화: 공통 PC 골격(S-n) 헬퍼 ──────────────────────────────

def _skeleton_id_map(base_pc_list: list[dict]) -> dict[str, str]:
    """{'S-1': <골격 PC 정본 이름>, ...}"""
    return _make_id_map([pc["pc"] for pc in base_pc_list], "S")


def _format_skeleton(base_pc_list: list[dict], ec_name_to_id: dict[str, str]) -> str:
    """골격 PC 목록을 S-n ID + 담당 EC ID로 표시"""
    return "\n".join(
        f'  S-{i}: "{pc["pc"]}"  ← covers: '
        + (", ".join(ec_name_to_id.get(e, "?") for e in pc.get("covers", [])) or "(none)")
        for i, pc in enumerate(base_pc_list, 1)
    )


def _apply_skeleton_origin(pc_list: list[dict], skel_ids: dict[str, str]) -> None:
    """origin("S-n" | "NEW")을 해석해 제자리 정규화 — 유지 PC는 골격 이름 그대로 복원.

    origin이 골격 ID이거나 이름이 골격과 정확히 같으면 "common", 그 외는 "model_specific".
    같은 골격 PC를 두 번 유지하면 두 번째부터는 model_specific으로 본다.
    """
    skel_names = set(skel_ids.values())
    used: set[str] = set()
    for pc in pc_list:
        origin = str(pc.get("origin", "") or "").strip()
        name = _resolve_quiet(origin, skel_ids) if origin else None
        if name is None and pc.get("pc") in skel_names:
            name = pc["pc"]
        if name and name not in used:
            pc["pc"] = name
            pc["origin"] = "common"
            used.add(name)
        else:
            pc["origin"] = "model_specific"


def _specialization_block(
    state: QFDState, base_pcs: list[dict], ec_name_to_id: dict[str, str],
) -> str:
    """PC 특화 블록 — Engineering(모든 라운드)·Market 공통. 첫 라운드는 아직 제안이 없음을 표시"""
    eng_msgs = [m for m in state.get("messages", []) if m["agent"] == "engineering_expert"]
    if eng_msgs:
        last = eng_msgs[-1]["content"]
        removed = last.get("removed", []) if isinstance(last, dict) else []
        diff_text = _format_skeleton_diff(state.get("pc_list", []), base_pcs, removed)
    else:
        diff_text = "  (no proposal yet — build the first one from the skeleton)"
    return PC_SPECIALIZATION_BLOCK.format(
        model_context=state.get("model_context", ""),
        skeleton_text=_format_skeleton(base_pcs, ec_name_to_id),
        diff_text=diff_text,
    )


def _format_skeleton_diff(
    pc_list: list[dict], base_pc_list: list[dict], removed_reasons: list[dict],
) -> str:
    """현재 제안과 골격의 차이 (유지 수 · 추가 PC와 사유 · 제외 PC와 사유)"""
    skel_ids = _skeleton_id_map(base_pc_list)
    name_to_sid = {v: k for k, v in skel_ids.items()}
    reason_by_sid: dict[str, str] = {}
    for r in removed_reasons or []:
        if isinstance(r, dict):
            sid = _resolve_quiet(str(r.get("id", "")), skel_ids)
            if sid:
                reason_by_sid[sid] = str(r.get("reason", ""))
    current = {pc["pc"] for pc in pc_list}
    kept = [n for n in skel_ids.values() if n in current]
    added = [pc for pc in pc_list if pc.get("origin") == "model_specific"]
    removed = [n for n in skel_ids.values() if n not in current]
    lines = [f"  KEPT from skeleton: {len(kept)}/{len(skel_ids)}"]
    lines += [f'  ADDED: "{pc["pc"]}" — reason: {pc.get("reason", "(none given)")}' for pc in added]
    lines += [f'  REMOVED: {name_to_sid[n]} "{n}" — reason: {reason_by_sid.get(n, "(none given)")}'
              for n in removed]
    return "\n".join(lines)


def engineering_expert_pc_node(state: QFDState) -> dict:
    """Engineering Expert: PC 리스트 제안/수정 (도구 호출 가능)

    발송 9 #22: PC마다 담당 EC ID(covers)를 제출받아 빈 행(미커버 EC)을 라운드 안에서
    검사한다 — Stage 1 EC 토론의 CR 커버리지 검사와 동형.
    """
    ec_names = [ec["ec"] for ec in state["ec_list"]]
    ec_ids = _make_id_map(ec_names, "EC")
    ec_name_to_id = {v: k for k, v in ec_ids.items()}
    ti = state.get("technical_importance", {})
    ec_text = "\n".join(
        f'  EC-{i}: "{ec["ec"]}" (TI: {ti.get(ec["ec"], "?")})'
        for i, ec in enumerate(state["ec_list"], 1)
    )
    min_pcs, max_pcs = _compute_pc_range(state)
    prev_gaps = state.get("coverage_gaps", []) or []
    gap_block = _format_gap_block(
        prev_gaps, ec_name_to_id, "EC", "PC",
        extra={g: f"TI {ti.get(g, 0):.1f}" for g in prev_gaps},
    )

    if state["iteration"] == 0:
        user_content = f"""Propose Part Characteristics (PCs) for the following ECs.
PCs are concrete component-level specifications that implement the ECs.

### Engineering Characteristics with Technical Importance:
{ec_text}

### Task:
FIRST: Use your search tools to research real laptop component specifications.
THEN: Propose PCs that collectively cover ALL ECs. Each PC must be a SINGLE
measurable parameter at the component/subsystem level.

### Count Constraint (STRICT):
Propose between {min_pcs} and {max_pcs} PCs in total. Every PC must be traceable
to at least one EC — if a PC cannot be linked to any EC, do not propose it.
If two PCs implement the SAME EC through the SAME physical mechanism, merge them
into one. Prefer the PC that is measurable at the component level.

### Coverage — no empty rows:
For each PC, list in "covers" the EC IDs (e.g. "EC-3") it primarily implements.
Every EC ID above must appear in at least one PC's "covers". If the PC limit
makes that impossible, list the exceptions in "uncovered_ecs" with a reason.

Output JSON:
{{
    "pc_list": [{{"pc": "parameter name (unit)", "covers": ["EC-3", "EC-7"]}}],
    "uncovered_ecs": [{{"ec": "EC-n", "reason": "..."}}],
    "reasoning": "How the PC set covers all ECs"
}}"""
    else:
        feedback = [m for m in state["messages"] if m["agent"] == "market_expert"][-1]
        cur_pcs = state.get("pc_list", [])
        cur_pc_text = "\n".join(
            f'  PC-{i}: "{pc["pc"]}"  ← covers: '
            + (", ".join(ec_name_to_id.get(e, "?") for e in pc.get("covers", [])) or "(none)")
            for i, pc in enumerate(cur_pcs, 1)
        )
        user_content = f"""The Market Expert reviewed your PC proposal:

{json.dumps(feedback["content"], indent=2, ensure_ascii=False)}

### Engineering Characteristics (ECs) with Technical Importance:
{ec_text}

### Your current PC list ({len(cur_pcs)}):
{cur_pc_text}

{gap_block}
Revise THIS list according to the feedback — do not rebuild it from scratch.
Keep every PC you do not explicitly change, and output the full revised list.

Address each concern with evidence. Accept valid market feedback,
reject points that conflict with engineering requirements.

Stay within {min_pcs}–{max_pcs} PCs total. If the count exceeds {max_pcs}, merge
PCs that implement the same EC through the same physical mechanism, and drop any
PC that is not traceable to an EC.

Coverage: for each PC list in "covers" the EC IDs it primarily implements. Every EC ID
must appear in at least one PC's "covers" — an EC with no PC is an empty matrix row.
Set "finalized": true ONLY when all feedback is addressed AND every EC is covered
(or each exception is justified in "uncovered_ecs").

Output JSON:
{{
    "pc_list": [{{"pc": "parameter name (unit)", "covers": ["EC-3", "EC-7"]}}],
    "uncovered_ecs": [{{"ec": "EC-n", "reason": "..."}}],
    "feedback_responses": [{{"feedback": "original", "action": "ACCEPT|REJECT", "detail": "..."}}],
    "reasoning": "Summary of changes",
    "finalized": true | false
}}"""

    # Separation 제품군 특화: 공통 골격이 주어지면 특화 블록(모델 맥락 · 골격 · 현재 차이 · 규칙)을 덧붙인다
    base_pcs = state.get("base_pc_list") or []
    skel_ids = _skeleton_id_map(base_pcs) if base_pcs else {}
    if base_pcs:
        user_content += "\n\n" + _specialization_block(state, base_pcs, ec_name_to_id)

    llm_with_tools = _chat_llm.bind_tools(ENGINEERING_TOOLS)
    messages = [
        {"role": "system", "content": ENGINEERING_EXPERT_PC_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 6
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [eng/pc] {tool_name}({list(tool_args.values())[0][:60]}...)")
            tool_map = {t.name: t for t in ENGINEERING_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [eng/pc] 도구 {tool_call_count}회 호출 완료")

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(ENGINEERING_EXPERT_PC_SYSTEM, user_content)

    pc_list = result.get("pc_list", [])
    for pc in pc_list:
        pc.pop("related_ecs", None)  # 구 필드 (하위 호환)
        if isinstance(pc.get("pc"), str):   # 표시용 "PC-n: " 접두 제거 (EC 노드와 동형)
            pc["pc"] = re.sub(r"^\s*PC-\d+\s*:\s*", "", pc["pc"]).strip()
    if base_pcs:
        _apply_skeleton_origin(pc_list, skel_ids)
        n_common = sum(1 for pc in pc_list if pc["origin"] == "common")
        print(f"    [PC 특화] 골격 유지 {n_common}/{len(base_pcs)} · "
              f"모델 전용 {len(pc_list) - n_common}")

    # 빈 행 검사 (발송 9 #22): covers ID → 정본 EC 이름, 미커버 EC 산출
    gaps, n_nocov = _coverage_gaps(pc_list, "covers", ec_ids)
    if n_nocov:
        print(f"    ⚠️  covers 필드 없는 PC {n_nocov}개")
    if gaps:
        print(f"    ⚠️  미커버 EC {len(gaps)}개 (빈 행): "
              + ", ".join(ec_name_to_id.get(g, "?") for g in gaps))
    else:
        print(f"    ✅ 커버리지: EC {len(ec_ids)}개 전부 PC 배정")

    update = {
        "pc_list": pc_list,
        "coverage_gaps": gaps,
        "iteration": state["iteration"] + 1,
        "messages": [
            {
                "agent": "engineering_expert",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }

    if state["iteration"] > 0:
        if state["iteration"] + 1 >= MIN_PC_DEBATE_ROUNDS:
            if gaps:
                print(f"    ⚠️  미커버 EC {len(gaps)}개 — finalized 거부, 추가 라운드")
                update["consensus"] = False
            else:
                update["consensus"] = result.get("finalized", False)
        else:
            update["consensus"] = False

    return update


def market_expert_pc_node(state: QFDState) -> dict:
    """Market Expert: PC 리스트 검토 (도구 호출 가능) — covers 표와 미커버 EC 목록 제공"""
    ec_names = [ec["ec"] for ec in state["ec_list"]]
    ec_ids = _make_id_map(ec_names, "EC")
    ec_name_to_id = {v: k for k, v in ec_ids.items()}
    ti = state.get("technical_importance", {})
    ec_text = "\n".join(
        f'  EC-{i}: "{ec["ec"]}" (TI: {ti.get(ec["ec"], "?")})'
        for i, ec in enumerate(state["ec_list"], 1)
    )
    pc_text = "\n".join(
        f'  PC-{i}: "{pc["pc"]}"  ← covers: '
        + (", ".join(ec_name_to_id.get(e, "?") for e in pc.get("covers", [])) or "(none)")
        for i, pc in enumerate(state.get("pc_list", []), 1)
    )
    gaps = state.get("coverage_gaps", []) or []
    uncovered_text = (
        "### ECs covered by NO PC (empty rows — Engineering's own mapping):\n"
        + "\n".join(f'  - {ec_name_to_id.get(g, "?")}: "{g}" (TI {ti.get(g, 0):.1f})' for g in gaps) + "\n"
        if gaps else
        "### ECs covered by NO PC: none (every EC has at least one PC)\n"
    )

    min_pcs, max_pcs = _compute_pc_range(state)

    user_content = f"""Review the Engineering Expert's PC proposal.

### ECs:
{ec_text}

### Proposed PCs ({len(state.get('pc_list', []))}):
{pc_text}

{uncovered_text}
Use search tools to check competitor product specifications and market standards.
For every EC listed as covered by NO PC, name it by ID in "uncovered_ecs" and suggest a PC.

### Count Check:
The agreed range is {min_pcs}–{max_pcs} PCs. The current proposal has
{len(state.get('pc_list', []))}. If it exceeds {max_pcs}, list the redundant ones in
"unnecessary_pcs" — prioritize PCs that duplicate another PC's physical mechanism
for the same EC, and PCs not traceable to any EC.

Output JSON:
{{
    "feedback_summary": "1-2 sentence assessment",
    "pc_feedback": [{{"pc": "name", "issue": "description", "suggestion": "change", "evidence": "market data"}}],
    "missing_pcs": ["PC that should be added"],
    "unnecessary_pcs": ["PC that is too theoretical"],
    "uncovered_ecs": ["EC id with no PC — and the PC you suggest for it"]
}}"""

    # Separation 제품군 특화: 같은 특화 블록으로 골격 대비 추가·제외와 그 사유를 검토
    base_pcs = state.get("base_pc_list") or []
    if base_pcs:
        user_content += "\n\n" + _specialization_block(state, base_pcs, ec_name_to_id)

    llm_with_tools = _chat_llm.bind_tools(MARKET_TOOLS)
    messages = [
        {"role": "system", "content": MARKET_EXPERT_PC_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 6
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"    🔧 [mkt/pc] {tool_name}({list(tool_args.values())[0][:60]}...)")
            tool_map = {t.name: t for t in MARKET_TOOLS}
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [mkt/pc] 도구 {tool_call_count}회 호출 완료")

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(MARKET_EXPERT_PC_SYSTEM, user_content)

    return {
        "messages": [
            {
                "agent": "market_expert",
                "round": state["iteration"],
                "content": result,
            }
        ],
    }


def pc_consensus_check(state: QFDState) -> str:
    if state["iteration"] >= settings.MAX_DEBATE_ROUNDS or state.get("consensus", False):
        return "end"
    return "continue"


def build_pc_debate_graph():
    """Stage 3: PC 리스트 토론 그래프"""
    graph = StateGraph(QFDState)
    graph.add_node("engineering_expert", engineering_expert_pc_node)
    graph.add_node("market_expert", market_expert_pc_node)
    graph.set_entry_point("engineering_expert")
    graph.add_conditional_edges("engineering_expert", pc_consensus_check, {
        "continue": "market_expert",
        "end": END,
    })
    graph.add_edge("market_expert", "engineering_expert")
    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: PC Roof Debate (Phase 2-B)
# ═════════════════════════════════════════════════════════════════════════════

def _generate_pc_pairs(pc_list: list[dict]) -> str:
    """PC 리스트에서 모든 고유 쌍 목록을 PC-n ID로 생성 (Stage 1 EC 쌍과 동형)"""
    n = len(pc_list)
    lines = []
    idx = 1
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"  {idx}. PC-{i + 1} ↔ PC-{j + 1}")
            idx += 1
    return "\n".join(lines)


def _format_pcs_ids(pc_list: list[dict]) -> str:
    """PC를 'PC-n' ID + 이름으로 포맷"""
    return "\n".join(f'PC-{i}: "{pc["pc"]}"' for i, pc in enumerate(pc_list, 1))


def engineering_expert_pc_roof_node(state: QFDState) -> dict:
    """Engineering Expert: PC Roof Matrix 제안/수정 (Stage 1 Roof와 동형 — PC-n ID 키)"""
    pc_list = state.get("pc_list", [])
    pc_text = _format_pcs_ids(pc_list)
    pc_ids = _make_id_map([pc["pc"] for pc in pc_list], "PC")
    name_to_id = {v: k for k, v in pc_ids.items()}
    pc_pairs_text = _generate_pc_pairs(pc_list)
    n_pcs = len(pc_list)
    n_pairs = n_pcs * (n_pcs - 1) // 2

    if state["iteration"] == 0:
        user_content = f"""Build the Roof Matrix (PC × PC correlation) for the confirmed PC set.

### Confirmed PCs ({n_pcs}):
{pc_text}

### All PC Pairs to Evaluate ({n_pairs} pairs):
{pc_pairs_text}

For EACH pair, assign a value if there is physical coupling:
  -0.9 (strong negative) | -0.5 (weak negative) | 0.5 (weak positive) | 0.9 (strong positive)
Omit pairs with no coupling.

Use the PC IDs shown above (e.g. "PC-3") as JSON keys.
IMPORTANT: Answer with IDs ONLY — do NOT retype PC names.

Output JSON:
{{
    "roof_matrix": {{"<PC id>": {{"<PC id>": <-0.9|-0.5|0.5|0.9>}}}},
    "reasoning": "Key correlations and their engineering basis"
}}"""
    else:
        feedback = [m for m in state["messages"] if m["agent"] == "market_expert"][-1]
        cur_roof = state.get("roof_matrix", {})
        cur_roof_text = _format_roof_lines(cur_roof, name_to_id)
        n_cur = sum(len(r) for r in cur_roof.values() if isinstance(r, dict))
        user_content = f"""The Market Expert reviewed your PC Roof Matrix:

{json.dumps(feedback["content"], indent=2, ensure_ascii=False)}

### Your current Roof Matrix ({n_cur} entries):
{cur_roof_text}

Revise THIS matrix: keep every pair you do not explicitly change, and add or
modify only the pairs raised in the feedback (or that you now judge wrong).
If you remove a pair, say why in feedback_responses. Output the FULL matrix.

Allowed values: {{-0.9, -0.5, 0.5, 0.9}}

### PCs ({n_pcs}):
{pc_text}

Use the PC IDs above (e.g. "PC-3") as JSON keys.
IMPORTANT: Answer with IDs ONLY — do NOT retype PC names.

Output JSON:
{{
    "roof_matrix": {{"<PC id>": {{"<PC id>": <-0.9|-0.5|0.5|0.9>}}}},
    "feedback_responses": [{{"concern": "original", "action": "ACCEPT|REJECT", "detail": "..."}}],
    "reasoning": "Summary of changes",
    "finalized": true | false
}}"""

    llm_with_tools = _chat_llm.bind_tools(ENGINEERING_TOOLS)
    messages = [
        {"role": "system", "content": ENGINEERING_EXPERT_PC_ROOF_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_map = {t.name: t for t in ENGINEERING_TOOLS}
            tool_name = tc["name"]
            print(f"    🔧 [eng/pc_roof] {tool_name}({list(tc['args'].values())[0][:60]}...)")
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(ENGINEERING_EXPERT_PC_ROOF_SYSTEM, user_content)

    result = _validate_roof(result)

    # ID → 정본 PC 이름 치환 + 키 검증 (해석 불가 키는 경고 후 폐기) — Stage 1 Roof와 동일
    resolved_roof = _resolve_matrix_ids(
        result.get("roof_matrix", {}), pc_ids, pc_ids, "PC Roof"
    )
    result["roof_matrix"] = resolved_roof
    print(f"    [PC Roof] PC 키 확정: {len(resolved_roof)}개 행 (정본 대조 완료)")

    update = {
        "roof_matrix": result.get("roof_matrix", {}),
        "iteration": state["iteration"] + 1,
        "messages": [
            {
                "agent": "engineering_expert",
                "round": state["iteration"] + 1,
                "content": result,
            }
        ],
    }

    if state["iteration"] > 0:
        if state["iteration"] + 1 >= MIN_PC_ROOF_DEBATE_ROUNDS:
            update["consensus"] = result.get("finalized", False)
        else:
            update["consensus"] = False

    return update


def market_expert_pc_roof_node(state: QFDState) -> dict:
    """Market Expert: PC Roof Matrix 검토 (PC-n ID 표기)"""
    pc_list = state.get("pc_list", [])
    pc_text = _format_pcs_ids(pc_list)
    pc_ids = _make_id_map([pc["pc"] for pc in pc_list], "PC")
    name_to_id = {v: k for k, v in pc_ids.items()}
    pc_pairs_text = _generate_pc_pairs(pc_list)
    n_pcs = len(pc_list)
    n_pairs = n_pcs * (n_pcs - 1) // 2

    existing_pairs = set()
    for a in state.get("roof_matrix", {}):
        for b in state["roof_matrix"][a]:
            existing_pairs.add(tuple(sorted([a, b])))

    user_content = f"""Review the PC Roof Matrix.

### PCs ({n_pcs}):
{pc_text}

### All Possible PC Pairs ({n_pairs} total):
{pc_pairs_text}

### Current Roof Matrix ({len(existing_pairs)} pairs assigned out of {n_pairs}):
{_format_roof_lines(state.get("roof_matrix", {}), name_to_id)}

Refer to PCs by their IDs (e.g. "PC-3") in your concerns.

Output JSON:
{{
    "feedback_summary": "1-2 sentence assessment",
    "coverage_assessment": "Are key trade-offs captured?",
    "roof_concerns": [{{"pc_a": "<PC id>", "pc_b": "<PC id>", "issue": "description", "evidence": "data"}}]
}}"""

    llm_with_tools = _chat_llm.bind_tools(MARKET_TOOLS)
    messages = [
        {"role": "system", "content": MARKET_EXPERT_PC_ROOF_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    max_tool_calls = 4
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_map = {t.name: t for t in MARKET_TOOLS}
            tool_name = tc["name"]
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    try:
        result = _parse_json_from_ai(response)
    except json.JSONDecodeError:
        result = _llm_json(MARKET_EXPERT_PC_ROOF_SYSTEM, user_content)

    return {
        "messages": [
            {
                "agent": "market_expert",
                "round": state["iteration"],
                "content": result,
            }
        ],
    }


def pc_roof_consensus_check(state: QFDState) -> str:
    if state["iteration"] >= settings.MAX_DEBATE_ROUNDS or state.get("consensus", False):
        return "end"
    return "continue"


def build_pc_roof_debate_graph():
    """Stage 3: PC Roof Matrix 토론 그래프"""
    graph = StateGraph(QFDState)
    graph.add_node("engineering_expert", engineering_expert_pc_roof_node)
    graph.add_node("market_expert", market_expert_pc_roof_node)
    graph.set_entry_point("engineering_expert")
    graph.add_conditional_edges("engineering_expert", pc_roof_consensus_check, {
        "continue": "market_expert",
        "end": END,
    })
    graph.add_edge("market_expert", "engineering_expert")
    return graph.compile()


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: EC-PC Matrix Fill (Phase 2-C)
# ═════════════════════════════════════════════════════════════════════════════

def _validate_pc_relationship(result: dict) -> dict:
    """EC-PC 행렬 값을 {0, 3, 9}로 보정 (Stage 1 _validate_relationship과 동형)"""
    valid = {0, 3, 9}
    violations = 0
    rel = result.get("relationship_matrix", {})
    for ec in rel:
        if not isinstance(rel[ec], dict):
            continue
        for pc, v in rel[ec].items():
            if v is None or not isinstance(v, (int, float)):
                rel[ec][pc] = 0
                violations += 1
            elif v not in valid:
                rel[ec][pc] = _snap_to_valid(abs(v), valid)
                violations += 1
    if violations:
        print(f"  ⚠️  EC-PC: {violations}개 값 보정됨 ({{0,3,9}} 외)")
    return result


def _build_pc_prev_context(accumulated_rel: dict, name_to_id: dict[str, str]) -> str:
    """이전 청크에서 확정된 EC의 주요 PC(9)를 요약 — 청크 간 일관성용"""
    if not accumulated_rel:
        return ""
    lines = ["### Previously completed ECs (values fixed — keep consistent):"]
    for ec, row in accumulated_rel.items():
        primaries = [name_to_id.get(pc, pc) for pc, v in row.items() if v == 9]
        lines.append(f'  - "{ec[:60]}": primary PCs {primaries if primaries else "none"}')
    return "\n".join(lines)


def fill_ec_pc_matrix(
    ec_list: list[dict],
    pc_list: list[dict],
    technical_importance: dict[str, float],
    hoq_label: str = "integrated",
) -> dict:
    """EC-PC Relationship Matrix — Engineering ↔ Market 토론으로 채움
    (발송 9 #19: Stage 1 fill_segment_matrix와 동형 — 매뉴얼 3-2-4)

    각 Chunk (5 ECs)마다 최대 MAX_DEBATE_ROUNDS 라운드:
      Round N:
        1) Engineering Expert  Phase 1 (CoT) → Phase 2 (JSON proposal)
        2) Market Expert       Phase 1 (CoT) → Phase 2 (JSON verdict)
      Market approved → 다음 Chunk / MAX_ROUNDS 도달 → 마지막 제안 채택

    Returns:
        {"relationship_matrix": {ec: {pc: 0|3|9}},
         "_meta": {"chunks": int, "total_rounds": int, "chunk_consensus": [bool]}}
    """
    pc_names = [pc["pc"] for pc in pc_list]
    pc_ids = _make_id_map(pc_names, "PC")
    pc_name_to_id = {v: k for k, v in pc_ids.items()}
    # 10/1: PC 토론(3-2-2)의 담당 선언(covers)을 프롬프트에 표시 (10/11: 하한 강제는 제거, 관찰만)
    ec_name_to_id_all = {ec["ec"]: f"EC-{i}" for i, ec in enumerate(ec_list, 1)}
    declared: dict[str, set[str]] = {}      # EC 이름 → 담당 선언 PC 이름 집합
    for pc in pc_list:
        for ec_name in pc.get("covers", []) or []:
            if ec_name in ec_name_to_id_all:
                declared.setdefault(ec_name, set()).add(pc["pc"])
    pc_text = "\n".join(
        f'PC-{i}: "{pc["pc"]}"'
        + (f'  ← declared for: {", ".join(ec_name_to_id_all[e] for e in pc.get("covers", []) if e in ec_name_to_id_all)}'
           if any(e in ec_name_to_id_all for e in (pc.get("covers", []) or [])) else "")
        for i, pc in enumerate(pc_list, 1)
    )
    max_rounds = settings.MAX_DEBATE_ROUNDS

    # EC를 chunk로 분할
    chunks = [
        ec_list[i : i + PC_MATRIX_FILL_CHUNK_SIZE]
        for i in range(0, len(ec_list), PC_MATRIX_FILL_CHUNK_SIZE)
    ]
    total_chunks = len(chunks)
    print(
        f"  [EC-PC Matrix] Fill 토론 시작 — "
        f"{len(ec_list)} ECs ÷ {PC_MATRIX_FILL_CHUNK_SIZE} = {total_chunks} chunks, "
        f"max {max_rounds} rounds/chunk"
    )

    accumulated_rel: dict = {}
    fill_log_messages: list[dict] = []
    total_debate_rounds = 0
    chunk_consensus_log: list[bool] = []

    for chunk_idx, chunk_ecs in enumerate(chunks, 1):
        chunk_start = (chunk_idx - 1) * PC_MATRIX_FILL_CHUNK_SIZE + 1
        ec_chunk_text = _format_ecs_ids(
            chunk_ecs, technical_importance, start=chunk_start
        )
        ec_ids = _make_id_map(
            [ec["ec"] for ec in chunk_ecs], "EC", start=chunk_start
        )
        prev_context = _build_pc_prev_context(accumulated_rel, pc_name_to_id)

        chunk_consensus = False
        eng_proposal: dict = {}
        eng_reasoning_text = ""
        mkt_reasoning_text = ""
        mkt_adjustments: list[dict] = []

        for debate_round in range(1, max_rounds + 1):
            total_debate_rounds += 1

            # ══ Engineering Expert — Phase 1: CoT Reasoning ══
            if debate_round == 1:
                eng_user_msg = f"""Analyze the EC-PC relationships.
(Chunk {chunk_idx}/{total_chunks})

### Part Characteristics (PCs):
{pc_text}

### Engineering Characteristics (ECs) to analyze in this chunk (with TI):
{ec_chunk_text}

{prev_context}

### Task:
For EACH EC, analyze its relationship with EVERY PC:
1. What does this EC require technically?
2. For each PC: Does this PC directly help achieve this EC?
3. Is it a primary driver (9), contributing factor (3), or no relation (0)?

Think step by step."""
            else:
                eng_user_msg = f"""The Market Expert reviewed your proposal and provided feedback.

### Market Expert's reasoning:
{mkt_reasoning_text}

### Market Expert's suggested adjustments:
{json.dumps(mkt_adjustments, indent=2, ensure_ascii=False)}

### Your previous proposal:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### ECs in this chunk:
{ec_chunk_text}

### PCs:
{pc_text}

{prev_context}

### Task:
Address EACH adjustment from the Market Expert with ACCEPT or REJECT:
- ACCEPT: Explain why, and state the revised value.
- REJECT: Explain with engineering reasoning why the original value is correct.

After addressing all feedback, provide your revised analysis for the full chunk."""

            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} "
                  f"Round {debate_round}/{max_rounds} — Eng. Expert CoT...")
            eng_reasoning_text = _llm_text(
                build_pc_matrix_fill_reasoning_system("engineering"), eng_user_msg
            )
            fill_log_messages.append({
                "agent": "engineering_expert_reasoning",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": eng_reasoning_text,
            })

            # ══ Engineering Expert — Phase 2: JSON ══
            eng_json_msg = f"""Convert your reasoning into a Relationship Matrix.

### Your reasoning analysis:
{eng_reasoning_text}

### ECs in this chunk:
{ec_chunk_text}

### PCs:
{pc_text}

### Rules:
- Follow your reasoning analysis faithfully.
- Allowed values ONLY: {{0, 3, 9}}
- Use the EC IDs and PC IDs shown above (e.g. "EC-7", "PC-3") as JSON keys.
  IMPORTANT: Answer with IDs ONLY — do NOT retype EC or PC names.
- Every EC-PC pair MUST have a value (use 0 if no connection).

Output JSON:
{{
    "relationship_matrix": {{"<EC id>": {{"<PC id>": <0|3|9>}}}}
}}"""

            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} "
                  f"Round {debate_round}/{max_rounds} — Eng. Expert JSON...")
            eng_result = _llm_json(
                build_pc_matrix_fill_json_system("engineering"), eng_json_msg
            )
            eng_result = _validate_pc_relationship(eng_result)
            eng_proposal = _resolve_matrix_ids(
                eng_result.get("relationship_matrix", {}), ec_ids, pc_ids, "EC-PC",
            )
            # 10/11: 담당 선언 쌍의 하한(0→3 강제)은 제거 — 판단은 에이전트의 몫.
            # 선언 쌍이 0으로 남은 수만 관찰값으로 기록한다.
            zero_declared = sum(
                1 for ec_name, row in eng_proposal.items()
                for pc_name in declared.get(ec_name, ()) if not row.get(pc_name)
            )
            if zero_declared:
                print(f"    ℹ️  [EC-PC] 담당 선언 쌍 {zero_declared}개가 0으로 남음 (관찰값, 강제 없음)")
            fill_log_messages.append({
                "agent": "engineering_expert_json",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": eng_proposal,
            })

            # ══ Market Expert — Phase 1: CoT Review ══
            mkt_user_msg = f"""Review the Engineering Expert's EC-PC relationship proposal.

### Engineering Expert's reasoning:
{eng_reasoning_text}

### Engineering Expert's proposed values:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### ECs in this chunk:
{ec_chunk_text}

### PCs:
{pc_text}

### Task:
Evaluate the proposal from a product/market perspective:
1. For each EC-PC pair: Is the magnitude appropriate (9 / 3 / 0)?
2. Are there PCs rated 0 that real products rely on for this EC?
3. Are there values rated 9 that are actually secondary levers?

Be specific about which values you agree or disagree with, and why."""

            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} "
                  f"Round {debate_round}/{max_rounds} — Mkt. Expert CoT...")
            mkt_reasoning_text = _llm_text(
                build_pc_matrix_fill_reasoning_system("market"), mkt_user_msg
            )
            fill_log_messages.append({
                "agent": "market_expert_reasoning",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": mkt_reasoning_text,
            })

            # ══ Market Expert — Phase 2: JSON Verdict ══
            mkt_json_msg = f"""Structure your review into a JSON verdict.

### Your review analysis:
{mkt_reasoning_text}

### Engineering Expert's current proposal:
{json.dumps(eng_proposal, indent=2, ensure_ascii=False)}

### ECs in this chunk:
{ec_chunk_text}

### PCs:
{pc_text}

### Rules:
- Set "approved": true if the proposal is acceptable overall.
- Set "approved": false if there are significant values that need change.
- Only list values you genuinely disagree with in "adjustments".
- Use the EC ID (e.g. "EC-7") for the "ec" field and the PC ID (e.g. "PC-3")
  for the "pc" field — do NOT retype names.
- Suggested values must be in {{0, 3, 9}}.

Output JSON:
{{
    "approved": true/false,
    "overall_assessment": "1-2 sentence summary",
    "adjustments": [
        {{"ec": "<EC id>", "pc": "<PC id>", "current": <value>, "suggested": <value>, "reason": "brief justification"}}
    ]
}}"""

            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} "
                  f"Round {debate_round}/{max_rounds} — Mkt. Expert JSON...")
            mkt_result = _llm_json(
                build_pc_matrix_fill_json_system("market"), mkt_json_msg
            )
            mkt_adjustments = mkt_result.get("adjustments", [])
            chunk_consensus = bool(mkt_result.get("approved", False))
            fill_log_messages.append({
                "agent": "market_expert_json",
                "round": chunk_idx,
                "debate_round": debate_round,
                "content": mkt_result,
            })

            status = "✅ approved" if chunk_consensus else f"❌ {len(mkt_adjustments)} adjustments"
            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} "
                  f"Round {debate_round}/{max_rounds} — {status}")

            if chunk_consensus:
                break

        # ── Chunk 결과 누적 (승인 또는 MAX 라운드 → 마지막 Eng 제안 채택) ──
        accumulated_rel.update(eng_proposal)
        chunk_consensus_log.append(chunk_consensus)
        if not chunk_consensus:
            print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} — 미승인 종료, "
                  f"Engineering 마지막 제안 채택")
        print(f"    [EC-PC] Chunk {chunk_idx}/{total_chunks} 완료 — "
              f"{len(eng_proposal)} ECs, {debate_round} rounds")

    # ID 치환이 라운드마다 수행되므로 여기서는 정본 일치만 확인
    matched = sum(1 for k in accumulated_rel if k in {e["ec"] for e in ec_list})
    print(f"  [EC-PC Matrix] Fill 완료 — {len(accumulated_rel)} ECs x {len(pc_list)} PCs, "
          f"총 {total_debate_rounds} debate rounds, "
          f"청크 승인 {sum(chunk_consensus_log)}/{total_chunks}")
    print(f"  [EC-PC Matrix] EC 라벨 일치: {matched}/{len(ec_list)}")
    # 10/11: 밀도는 규칙이 아니라 관찰값으로만 기록 (비0 비율, EC당 평균 PC 수)
    _cells = [v for row in accumulated_rel.values() for v in row.values()]
    _nz = sum(1 for v in _cells if v)
    if _cells:
        print(f"  [EC-PC Matrix] 관찰: 비0 {_nz}/{len(_cells)} ({100*_nz/len(_cells):.0f}%), "
              f"EC당 평균 {_nz/max(len(accumulated_rel),1):.1f} PCs")
    if matched < len(ec_list):
        print(f"  ⚠️  [EC-PC Matrix] {len(ec_list) - matched}개 EC가 매칭되지 않았습니다 — "
              f"해당 EC는 Part Importance 계산에서 제외됩니다.")

    save_debate_log(
        f"stage3_{hoq_label}_ec_pc_matrix_fill.txt",
        fill_log_messages,
        f"Stage 3 [{hoq_label}] — EC-PC Matrix Fill Debate "
        f"({len(ec_list)} ECs, {total_chunks} chunks, {total_debate_rounds} total rounds, "
        f"approved {sum(chunk_consensus_log)}/{total_chunks})",
    )

    return {
        "relationship_matrix": accumulated_rel,
        "_meta": {
            "chunks": total_chunks,
            "total_rounds": total_debate_rounds,
            "chunk_consensus": chunk_consensus_log,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Target Value Setting (Phase 2-D)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_final_reasoning(messages: list[dict], agent: str) -> str:
    """토론 메시지에서 해당 에이전트의 마지막 reasoning을 추출.

    별도 LLM 호출 없이 이미 기록된 최종 라운드 내용만 재사용한다
    (사양서 입력용 토론 결론)."""
    for m in reversed(messages):
        if m.get("agent") == agent and isinstance(m.get("content"), dict):
            reasoning = m["content"].get("reasoning", "")
            if reasoning:
                return str(reasoning)
    return ""


def _invoke_agent_with_tools(
    system: str, user: str, tools: list, tag: str, max_tool_calls: int = 8
) -> dict:
    """도구 바인딩 LLM 호출 → JSON 파싱 (Target Value 토론 공통 루프)"""
    llm_with_tools = _chat_llm.bind_tools(tools)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_call_count = 0
    tool_map = {t.name: t for t in tools}
    while response.tool_calls and tool_call_count < max_tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            print(f"    🔧 [{tag}] {tool_name}({list(tc['args'].values())[0][:60]}...)")
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            tool_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    if tool_call_count > 0:
        print(f"    📊 [{tag}] 도구 {tool_call_count}회 호출 완료")

    try:
        return _parse_json_from_ai(response)
    except json.JSONDecodeError:
        return _llm_json(system, user + "\n\nIMPORTANT: Output ONLY valid JSON.")


def set_target_values_via_agent(
    pc_list: list[dict],
    part_importance: dict[str, float],
    technical_importance: dict[str, float],
    ec_list: list[dict],
    ec_pc_matrix: dict[str, dict[str, int]],
    pc_roof_matrix: dict[str, dict[str, float]],
    hoq_label: str = "integrated",
    model_context: str | None = None,
) -> dict[str, dict]:
    """Engineering 제안 ⇄ Market 검토 토론으로 각 PC의 Target Value를 결정

    model_context가 주어지면(Separation) 공유 컨텍스트 앞에 설계 중인 모델 블록을 붙인다.

    Market이 승인(approved)하면 종료, 최대 라운드 소진 시 Engineering의
    마지막 제안을 채택한다 (fill_segment_matrix와 동일한 루프 패턴).

    Returns:
        {pc_name: {"value", "justification", "trade_off_notes"},
         "_meta": {"rounds": int, "consensus": bool}}
    """
    pc_names = [pc["pc"] for pc in pc_list]
    pc_ids = _make_id_map(pc_names, "PC")
    pc_canonical = set(pc_names)
    # 주석(PI)을 별도 줄로 분리 — 한 줄에 붙이면 LLM이 주석까지 키에 포함시킨다
    pc_text = "\n".join(
        f'PC-{i}: "{n}"\n   PI: {part_importance.get(n, 0):.1f}'
        for i, n in enumerate(pc_names, 1)
    )
    ec_text = _format_ecs_ids(ec_list, technical_importance)
    roof_text = json.dumps(pc_roof_matrix, indent=2, ensure_ascii=False)

    # EC-PC 관계 요약
    rel_lines = []
    for ec, pcs in ec_pc_matrix.items():
        nonzero = {pc: v for pc, v in pcs.items() if v > 0}
        if nonzero:
            rel_lines.append(f'  {ec}: {nonzero}')
    rel_text = "\n".join(rel_lines)

    shared_context = f"""### Part Characteristics (ranked by Part Importance):
{pc_text}

### Engineering Characteristics (for reference):
{ec_text}

### EC-PC Relationships (non-zero only):
{rel_text}

### PC Roof Matrix (trade-offs):
{roof_text}"""
    if model_context:
        shared_context = (TARGET_VALUE_MODEL_CONTEXT.format(model_context=model_context)
                          + "\n\n" + shared_context)

    max_rounds = settings.MAX_DEBATE_ROUNDS
    print(f"  [Target Value] 2-Agent 토론 시작 — "
          f"{len(pc_list)} PCs, max {max_rounds} rounds")

    target_values: dict[str, dict] = {}
    approved = False
    mkt_result: dict = {}
    log_messages: list[dict] = []
    rounds_used = 0

    for debate_round in range(1, max_rounds + 1):
        rounds_used = debate_round

        # ══ Engineering: 제안/수정 ══
        if debate_round == 1:
            eng_user = f"""Determine target values for each Part Characteristic.

{shared_context}

### Task:
For EACH PC above, determine a specific, quantitative target value.
FIRST: Use search tools to look up current laptop specifications and benchmarks.
THEN: Set targets that balance trade-offs and prioritize high-PI parts."""
        else:
            eng_user = f"""The Market Expert reviewed your target values.

### Market Expert's assessment:
{mkt_result.get("overall_assessment", "")}

### Market Expert's adjustments:
{json.dumps(mkt_result.get("adjustments", []), indent=2, ensure_ascii=False)}

### Your previous proposal:
{json.dumps(target_values, indent=2, ensure_ascii=False)}

{shared_context}

### Task:
Address EACH adjustment with ACCEPT or REJECT, then output your revised
FULL target value set (every PC, PC IDs as keys)."""

        print(f"    [Target Value] Round {debate_round}/{max_rounds} — "
              f"Engineering 제안...")
        eng_result = _invoke_agent_with_tools(
            TARGET_VALUE_ENGINEERING_SYSTEM, eng_user, ENGINEERING_TOOLS,
            "target/eng",
        )

        # PC ID → 정본 이름 치환 (해석 불가 키는 경고 후 폐기)
        raw_tv = eng_result.get("target_values", {})
        resolved: dict[str, dict] = {}
        for key, val in raw_tv.items():
            name = _resolve_key(key, pc_ids, pc_canonical, "TargetValue")
            if name is not None:
                resolved[name] = val
        if resolved:
            if debate_round > 1:
                kept = [n for n in target_values if n not in resolved]
                if kept:
                    print(f"    [Target Value] Round {debate_round} 부분 응답 — "
                          f"{len(kept)}개 PC는 이전 라운드 값 유지")
            # 덮어쓰기가 아니라 병합 — 부분 응답이 와도 나머지 목표값은 유지
            target_values = {**target_values, **resolved}

        log_messages.append({
            "agent": "target_value_engineering",
            "round": debate_round,
            "content": {**eng_result, "target_values": target_values},
        })

        # ══ Market: 검토 ══
        mkt_user = f"""Review the Engineering Expert's proposed target values.

### Proposed target values:
{json.dumps(target_values, indent=2, ensure_ascii=False)}

{shared_context}

### Task:
Evaluate each target value for market realism, competitiveness, trade-off
consistency, and priority alignment. Cite evidence for every adjustment."""

        print(f"    [Target Value] Round {debate_round}/{max_rounds} — "
              f"Market 검토...")
        mkt_result = _invoke_agent_with_tools(
            TARGET_VALUE_MARKET_SYSTEM, mkt_user, MARKET_TOOLS, "target/mkt",
        )
        approved = mkt_result.get("approved", False)
        adj_count = len(mkt_result.get("adjustments", []))

        log_messages.append({
            "agent": "target_value_market",
            "round": debate_round,
            "content": mkt_result,
        })

        status = "승인" if approved else f"조정 요구 {adj_count}건"
        print(f"    [Target Value] Round {debate_round}/{max_rounds} — {status}")

        if approved:
            break

    if not approved:
        print(f"  ⚠️  [Target Value] 최대 라운드 소진 (합의 실패) — "
              f"Engineering 마지막 제안 채택")

    # PC 누락 검사 — 조용히 통과시키지 않는다
    missing = pc_canonical - set(target_values.keys())
    if missing:
        print(f"  ⚠️  [Target Value] 목표값 누락 PC {len(missing)}개:")
        for pc in sorted(missing):
            print(f"      - {pc}")

    print(f"  [Target Value] 완료 — {len(target_values)}/{len(pc_list)} PCs, "
          f"{rounds_used} rounds, 합의={approved}")

    # 로그 저장 (Separation 시 덮어쓰기 방지를 위해 라벨 포함)
    save_debate_log(
        f"stage3_{hoq_label}_target_values.txt",
        log_messages,
        f"Stage 3 [{hoq_label}] — Target Value Debate "
        f"({len(pc_list)} PCs, {rounds_used} rounds, consensus={approved})",
    )

    # 라운드/합의 메타 — 매뉴얼 서술을 산출물로 검증하기 위해 기록
    # (하류 value 추출부는 "_meta" 키를 스킵함)
    target_values["_meta"] = {"rounds": rounds_used, "consensus": approved}
    return target_values


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Combined Part Deployment Runner
# PC Debate → PC Roof → EC-PC Matrix Fill → Part Importance → Target Value
# ═════════════════════════════════════════════════════════════════════════════

def run_pc_skeleton_debate(
    ec_list: list[dict],
    technical_importance: dict[str, float],
) -> list[dict]:
    """Separation 전용: 분할 전 전체 HoQ의 TI로 PC 토론을 1회 실행해 공통 PC 골격을 만든다.

    골격은 두 모델 PC 특화의 출발점이다 (Phase 2-A와 같은 토론 그래프 · 같은 프롬프트).
    """
    print(f"\n{'='*60}")
    print(f"  Part Deployment: 공통 PC 골격 (분할 전 전체 HoQ)")
    print(f"{'='*60}")
    pc_result = build_pc_debate_graph().invoke({
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "ec_list": ec_list,
        "technical_importance": technical_importance,
        "phase": "pc",
        "iteration": 0,
        "consensus": False,
        "messages": [],
    })
    pc_list = pc_result.get("pc_list", [])
    rounds = pc_result["iteration"]
    print(f"  [공통 골격] PC 토론 완료: {rounds}라운드, PC {len(pc_list)}개")
    save_debate_log(
        "stage3_common_skeleton_pc_debate.txt",
        pc_result["messages"],
        f"Stage 3 [common_skeleton] — PC Debate ({rounds} rounds, {len(pc_list)} PCs)",
    )
    return pc_list


def run_part_deployment(
    ec_list: list[dict],
    technical_importance: dict[str, float],
    hoq_label: str = "integrated",
    base_pc_list: list[dict] | None = None,
    model_context: str | None = None,
) -> dict:
    """하나의 Integrated HoQ에 대해 Part Deployment 전체 파이프라인 실행

    Separation이면 base_pc_list(공통 PC 골격)와 model_context(모델 맥락)를 받는다:
    PC 토론은 골격을 출발점으로 한 특화가 되고, Target Value 토론에 모델 맥락이 들어간다.
    둘 다 None이면(통합 1모델) 기존과 동일하게 동작한다.

    Returns:
        {
            "pc_list": [...],
            "ec_pc_relationship_matrix": {...},
            "pc_roof_matrix": {...},
            "part_importance": {...},
            "target_values": {...},
        }
    """
    print(f"\n{'='*60}")
    print(f"  Part Deployment: {hoq_label}")
    print(f"{'='*60}")

    # ── Phase 2-A: PC 리스트 토론 ──
    print(f"\n  [Phase 2-A] PC 리스트 토론 시작")
    pc_graph = build_pc_debate_graph()
    pc_state: QFDState = {
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "ec_list": ec_list,
        "technical_importance": technical_importance,
        "phase": "pc",
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }
    if base_pc_list:
        pc_state["base_pc_list"] = base_pc_list
        pc_state["model_context"] = model_context or ""
    pc_result = pc_graph.invoke(pc_state)

    pc_rounds = pc_result["iteration"]
    pc_list = pc_result.get("pc_list", [])
    pc_coverage_gaps = pc_result.get("coverage_gaps", []) or []
    pc_debate_summary = _extract_final_reasoning(
        pc_result["messages"], "engineering_expert"
    )
    print(f"  [Phase 2-A] PC 토론 완료: {pc_rounds}라운드, PC {len(pc_list)}개")
    if pc_coverage_gaps:
        print(f"  ⚠️  [Phase 2-A] 토론 종료 시 미커버 EC {len(pc_coverage_gaps)}개: "
              + "; ".join(g[:40] for g in pc_coverage_gaps))

    save_debate_log(
        f"stage3_{hoq_label}_pc_debate.txt",
        pc_result["messages"],
        f"Stage 3 [{hoq_label}] — PC Debate ({pc_rounds} rounds, {len(pc_list)} PCs)",
    )

    # ── Phase 2-B: PC Roof 토론 ──
    print(f"\n  [Phase 2-B] PC Roof Matrix 토론 시작")
    pc_roof_graph = build_pc_roof_debate_graph()
    roof_state: QFDState = {
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "ec_list": ec_list,
        "pc_list": pc_list,
        "phase": "pc_roof",
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }
    roof_result = pc_roof_graph.invoke(roof_state)

    roof_rounds = roof_result["iteration"]
    pc_roof_matrix = roof_result.get("roof_matrix", {})
    pc_roof_debate_summary = _extract_final_reasoning(
        roof_result["messages"], "engineering_expert"
    )
    print(f"  [Phase 2-B] PC Roof 토론 완료: {roof_rounds}라운드")

    save_debate_log(
        f"stage3_{hoq_label}_pc_roof_debate.txt",
        roof_result["messages"],
        f"Stage 3 [{hoq_label}] — PC Roof Debate ({roof_rounds} rounds)",
    )

    # ── Phase 2-C: EC-PC Matrix Fill ──
    print(f"\n  [Phase 2-C] EC-PC Matrix Fill 시작")
    matrix_result = fill_ec_pc_matrix(
        ec_list, pc_list, technical_importance, hoq_label=hoq_label
    )
    ec_pc_matrix = matrix_result["relationship_matrix"]

    # 빈 행 사후 검출 (발송 9 #22): 실제 행렬에서 어떤 PC와도 관계 0인 EC
    empty_ec_rows = [
        ec["ec"] for ec in ec_list
        if not any(ec_pc_matrix.get(ec["ec"], {}).get(pc["pc"], 0) for pc in pc_list)
    ]
    if empty_ec_rows:
        print(f"  ⚠️  [EC-PC Matrix] 빈 EC 행 {len(empty_ec_rows)}개 (어떤 PC와도 관계 0): "
              + "; ".join(e[:40] for e in empty_ec_rows))
    else:
        print(f"  ✅ [EC-PC Matrix] 빈 EC 행 없음")

    # ── Part Importance 계산: PI(PC) = Σ(TI(EC) × importance(EC, PC)) ──
    print(f"\n  [Phase 2-D] Part Importance 산출")
    part_importance: dict[str, float] = {}
    for pc in pc_list:
        pc_name = pc["pc"]
        pi = 0.0
        for ec in ec_list:
            ec_name = ec["ec"]
            ti = technical_importance.get(ec_name, 0.0)
            imp = ec_pc_matrix.get(ec_name, {}).get(pc_name, 0)
            pi += ti * imp
        part_importance[pc_name] = round(pi, 2)

    # PI 순으로 정렬 출력
    sorted_pi = sorted(part_importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (pc_name, pi) in enumerate(sorted_pi, 1):
        print(f"    {rank}. {pc_name}: {pi:.1f}")

    # ── Phase 2-E: Target Value 설정 ──
    print(f"\n  [Phase 2-E] Target Value 설정")
    target_values_raw = set_target_values_via_agent(
        pc_list, part_importance, technical_importance,
        ec_list, ec_pc_matrix, pc_roof_matrix, hoq_label=hoq_label,
        model_context=model_context if base_pc_list else None,
    )

    # value만 추출 (간결한 출력용) — "_meta"(라운드/합의 기록)는 스킵
    target_values_simple: dict[str, str] = {}
    for pc_name, tv in target_values_raw.items():
        if pc_name == "_meta":
            continue
        if isinstance(tv, dict):
            target_values_simple[pc_name] = tv.get("value", str(tv))
        else:
            target_values_simple[pc_name] = str(tv)

    # Separation 제품군 특화 기록: 골격 대비 유지·추가·제외와 사유 (마지막 Engineering 제안 기준)
    specialization: dict = {}
    if base_pc_list:
        eng_msgs = [m for m in pc_result["messages"] if m["agent"] == "engineering_expert"]
        last = eng_msgs[-1]["content"] if eng_msgs else {}
        current = {pc["pc"] for pc in pc_list}
        removed_reasons = {
            _resolve_quiet(str(r.get("id", "")), _skeleton_id_map(base_pc_list)): r.get("reason", "")
            for r in (last.get("removed", []) or []) if isinstance(r, dict)
        }
        specialization = {
            "skeleton_size": len(base_pc_list),
            "kept": [pc["pc"] for pc in pc_list if pc.get("origin") == "common"],
            "added": [{"pc": pc["pc"], "reason": pc.get("reason", "")}
                      for pc in pc_list if pc.get("origin") == "model_specific"],
            "removed": [{"pc": b["pc"], "reason": removed_reasons.get(b["pc"], "")}
                        for b in base_pc_list if b["pc"] not in current],
        }
        print(f"  [PC 특화] 유지 {len(specialization['kept'])} · 추가 {len(specialization['added'])} · "
              f"제외 {len(specialization['removed'])} (골격 {len(base_pc_list)})")

    return {
        "pc_list": pc_list,
        "ec_pc_relationship_matrix": ec_pc_matrix,
        "pc_roof_matrix": pc_roof_matrix,
        "part_importance": part_importance,
        "target_values": target_values_simple,
        "target_values_detailed": target_values_raw,
        "pc_debate_summary": pc_debate_summary,
        "pc_roof_debate_summary": pc_roof_debate_summary,
        # 발송 9 #22 빈 행 검사 기록 (빈 열 = PI 0 PC 제거는 stage3 drop_zero_importance_pcs)
        "coverage": {
            "pc_debate_uncovered_ecs": pc_coverage_gaps,
            "empty_ec_rows": empty_ec_rows,
            "ec_pc_fill_meta": matrix_result.get("_meta", {}),
        },
        "specialization": specialization,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 Phase 3: PC Specification (부품 특성 사양서)
# ═════════════════════════════════════════════════════════════════════════════

def generate_pc_specification(
    deployment: dict,
    strategy: str,
    conflict_summaries: list[dict],
    hoq_label: str = "integrated",
) -> list[dict]:
    """Part Deployment 결과 + 전략 컨텍스트를 종합하여 PC Specification 생성

    Args:
        deployment: run_part_deployment() 반환값
        strategy: "Compromise" | "Selection" | "Separation"
        conflict_summaries: [{group_cr_a, group_cr_b, strategy, reasoning, debate_summary}, ...]
        hoq_label: Integrated HoQ 라벨

    Returns:
        [{"pc", "rationale"}, ...] — LLM이 생성한 설정 근거
    """
    pc_list = deployment["pc_list"]
    part_importance = deployment["part_importance"]
    target_values = deployment["target_values"]
    ec_pc_matrix = deployment["ec_pc_relationship_matrix"]
    target_values_detailed = deployment.get("target_values_detailed", {})
    pc_debate_summary = deployment.get("pc_debate_summary", "")
    pc_roof_debate_summary = deployment.get("pc_roof_debate_summary", "")

    pc_names = [pc["pc"] for pc in pc_list]
    pc_ids = _make_id_map(pc_names, "PC")
    pc_canonical = set(pc_names)

    pc_summary_lines = []
    for idx, pc in enumerate(pc_list, 1):
        pc_name = pc["pc"]
        pi = part_importance.get(pc_name, 0.0)
        tv = target_values.get(pc_name, "N/A")

        related = []
        for ec_name, pcs in ec_pc_matrix.items():
            score = pcs.get(pc_name, 0)
            if score > 0:
                related.append(f"{ec_name}({score})")

        related_str = ", ".join(related) if related else "None"
        pc_summary_lines.append(
            f'- PC-{idx}: "{pc_name}" — PI={pi:.2f}, Target={tv}, Related ECs=[{related_str}]'
        )
        # Target Value 토론이 남긴 근거를 PC별로 병기 (별도 줄 — 키 오염 방지)
        detail = target_values_detailed.get(pc_name, {})
        if isinstance(detail, dict):
            if detail.get("justification"):
                pc_summary_lines.append(
                    f"    Target justification: {detail['justification']}"
                )
            if detail.get("trade_off_notes"):
                pc_summary_lines.append(
                    f"    Trade-off notes: {detail['trade_off_notes']}"
                )

    # 10/10 (4-1′): 그룹별 CR을 Judge·토론 원문과 같은 순서(cr_a → cr_b → common)의
    # 그룹 접두 번호(G1 CR-3)로 제시 — 원문의 그룹 안 번호를 독자가 식별할 수 있게 한다.
    conflict_context_lines = []
    for gi, cs in enumerate(conflict_summaries, 1):
        cr_lines: list[str] = []
        n = 0
        for seg_label, key in ((settings.SEGMENT_A, "group_cr_a"),
                               (settings.SEGMENT_B, "group_cr_b"),
                               ("Common", "group_cr_common")):
            for cr in cs.get(key, []) or []:
                n += 1
                cr_lines.append(f"  G{gi} CR-{n} [{seg_label}]: {cr}")
        conflict_context_lines.append(
            f"### Conflict Group G{gi} — Strategy: {cs.get('strategy', 'N/A')}\n"
            f"CR IDs below are local to this group (numbering restarts in each "
            f"group); the Reasoning and Debate texts use these local IDs.\n"
            + "\n".join(cr_lines) + "\n"
            f"Reasoning: {cs.get('reasoning', 'N/A')}\n"
            f"Debate: {cs.get('debate_summary', 'N/A')}"
        )

    user_prompt = (
        f"## Strategy: {strategy}\n"
        f"## HoQ Label: {hoq_label}\n\n"
        f"## PC Summary (use the PC IDs, e.g. \"PC-3\", in the \"pc\" field — "
        f"do NOT retype names):\n" + "\n".join(pc_summary_lines) + "\n\n"
        f"## Conflict Resolution Context:\n"
        + ("\n\n".join(conflict_context_lines) if conflict_context_lines else "No conflict records.")
        + (f"\n\n## PC Debate Conclusion:\n{pc_debate_summary}"
           if pc_debate_summary else "")
        + (f"\n\n## PC Roof Debate Conclusion:\n{pc_roof_debate_summary}"
           if pc_roof_debate_summary else "")
    )

    # 입력 크기 실측용 — 상한·절단은 두지 않는다 (재실행 실측 후 판단)
    print(f"  [PC Spec] user prompt 총 {len(user_prompt):,}자")
    print(f"  [PC Spec] LLM 호출 — {len(pc_list)} PCs에 대한 설정 근거 생성")
    result = _llm_json(PC_SPEC_SYSTEM, user_prompt)

    specs = result.get("specifications", [])

    # PC ID → 정본 이름 치환 (해석 불가 키는 원문 유지 — stage3 매칭부가 폴백 처리)
    resolved_cnt = 0
    for spec in specs:
        key = spec.get("pc")
        if not key:
            continue
        name = _resolve_key(str(key), pc_ids, pc_canonical, "PCSpec")
        if name is not None:
            spec["pc"] = name
            resolved_cnt += 1
    print(f"  [PC Spec] PC 키 확정: {resolved_cnt}/{len(specs)} (ID → 정본 이름)")

    save_debate_log(
        f"stage3_{hoq_label}_pc_specification.txt",
        [{"agent": "pc_spec_synthesizer", "round": 1, "content": result}],
        f"Stage 3 [{hoq_label}] — PC Specification ({len(specs)} PCs)",
    )

    return specs
