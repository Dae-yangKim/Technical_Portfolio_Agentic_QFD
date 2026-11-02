# Stage 0: Reddit → Purpose Tagging → Segment-Level CR Extraction → Final Processing
#
# Input:  data/raw/reddit_reviews.xlsx
# Output: data/intermediate/tagged_reviews.xlsx
#         data/intermediate/phase2_cr_extraction.xlsx
#         data/intermediate/final_cr_output.xlsx
#         ChromaDB: data/chroma/chroma_db_review_{purpose}/
#                   data/chroma/chroma_db_cr_{purpose}/
#
# Phase 1: Tagging               (LangGraph — TaggingState)
# Phase 2: Segment-Level CR Extraction (single LLM call per segment)
# Phase 3: Final Processing       (LangGraph — FinalProcessingState)
#   Priority → ChromaDB → CandidateSearch → LLMClassify
import os, json
import pandas as pd
import numpy as np
import nest_asyncio

from typing import List, TypedDict
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import settings
from src import prompts

nest_asyncio.apply()

# ── LLM & Embeddings ─────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=settings.MODEL,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    service_tier = settings.SERVICE_TIER
)

embeddings = OpenAIEmbeddings()

PURPOSES = [settings.SEGMENT_A, settings.SEGMENT_B]

# ── ChromaDB ─────────────────────────────────────────────────────────────────

review_collections = {
    purpose: Chroma(
        collection_name=f"review_{purpose.lower()}",
        embedding_function=embeddings,
        persist_directory=f"{settings.CHROMA_BASE_PATH}/chroma_db_review_{purpose.lower()}",
        collection_metadata={"hnsw:space": "cosine"},
    )
    for purpose in PURPOSES
}

cr_collections = {
    purpose: Chroma(
        collection_name=f"cr_{purpose.lower()}",
        embedding_function=embeddings,
        persist_directory=f"{settings.CHROMA_BASE_PATH}/chroma_db_cr_{purpose.lower()}",
        collection_metadata={"hnsw:space": "cosine"},
    )
    for purpose in PURPOSES
}

# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Tagging
# ═════════════════════════════════════════════════════════════════════════════

class TaggingState(TypedDict):
    raw_review:       str
    is_valid:         bool
    product_type:     str
    primary_purpose:  str
    reasoning:        str
    purpose_signals:  List[str]
    evidence_snippet: str


def tagging_node(state: TaggingState) -> dict:
    try:
        response = llm.invoke([
            SystemMessage(content=prompts.TAGGING_SYSTEM),
            HumanMessage(content=f'Review: "{state["raw_review"]}"'),
        ])
        result = json.loads(
            response.content.replace("```json", "").replace("```", "").strip()
        )
        return {
            "is_valid":         result.get("is_valid", False),
            "product_type":     result.get("product_type", "unclear"),
            "primary_purpose":  result.get("primary_purpose", "None"),
            "reasoning":        result.get("reasoning", ""),
            "purpose_signals":  result.get("purpose_signals", []),
            "evidence_snippet": result.get("evidence_snippet", ""),
        }
    except Exception as e:
        return {
            "is_valid": False, "product_type": "unclear",
            "primary_purpose": "None", "reasoning": f"Error: {e}",
            "purpose_signals": [], "evidence_snippet": "",
        }


def _tagging_should_continue(state: TaggingState) -> str:
    return "continue" if state["is_valid"] else "end"


_tagging_workflow = StateGraph(TaggingState)
_tagging_workflow.add_node("Tagger", tagging_node)
_tagging_workflow.set_entry_point("Tagger")
_tagging_workflow.add_conditional_edges(
    "Tagger", _tagging_should_continue, {"continue": END, "end": END}
)
tagging_app = _tagging_workflow.compile()


def run_phase1(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: 병렬 태깅 실행, tagged DataFrame 반환"""
    print(f"\n[Phase 1] Tagging {len(raw_data)}개 리뷰...")

    def _run_tagging(args):
        i, review = args
        try:
            state = tagging_app.invoke({"raw_review": review})
            return {
                "Index":           i,
                "Original_Review": review,
                "Is_Valid":        state.get("is_valid", False),
                "Product_Type":    state.get("product_type", "unclear"),
                "Purpose":         state.get("primary_purpose", "None"),
                "Reasoning":       state.get("reasoning", ""),
                "Evidence":        state.get("evidence_snippet", ""),
            }
        except Exception as e:
            return {
                "Index": i, "Original_Review": review,
                "Is_Valid": False, "Product_Type": "unclear",
                "Purpose": "None", "Reasoning": f"Error: {e}", "Evidence": "",
            }

    results = [None] * len(raw_data)
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_tagging, (i, row)): i
            for i, row in enumerate(raw_data["Body"])
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="[Phase 1] Tagging"):
            result = future.result()
            results[result["Index"]] = result

    tagging_df = pd.DataFrame(results)
    print("\n[Phase 1 완료] Purpose 분포:")
    print(tagging_df[tagging_df["Is_Valid"] == True]["Purpose"].value_counts())
    return tagging_df


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Segment-Level CR Extraction
# ═════════════════════════════════════════════════════════════════════════════


def _single_extraction(purpose: str, indexed_reviews: str) -> list[dict]:
    """단일 LLM 호출로 세그먼트 CR 추출"""
    system_prompt = prompts.build_segment_cr_extraction_system(
        purpose, settings.MAX_CRS_PER_SEGMENT, settings.MIN_REVIEW_SUPPORT,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=indexed_reviews),
    ])
    result = json.loads(
        response.content.replace("```json", "").replace("```", "").strip()
    )
    return result.get("cr_candidates", [])


def _merge_duplicate_crs(purpose: str, all_crs: list[dict]) -> list[dict]:
    """청크별 추출 결과에서 중복 CR 병합"""
    cr_list_text = json.dumps(all_crs, ensure_ascii=False, indent=2)
    system_prompt = prompts.CHUNK_MERGE_SYSTEM.format(
        max_crs=settings.MAX_CRS_PER_SEGMENT,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=cr_list_text),
    ])
    result = json.loads(
        response.content.replace("```json", "").replace("```", "").strip()
    )
    return result.get("cr_candidates", [])


def _format_indexed_reviews(reviews: list[str], offset: int = 0) -> str:
    return "\n".join(f"[{offset + i}] {r}" for i, r in enumerate(reviews))


def _dedup_evidence_indices(crs: list[dict]) -> int:
    """각 CR의 evidence_review_indices 중복 제거 (순서 유지, 제자리 수정).

    발송 9 #12: 묶음 병합 프롬프트는 합집합을 요구하지만 LLM이 중복을 남기고,
    그 개수가 최소 지지·상위 N 컷·분위수 Priority·쿼터 정렬에 그대로 쓰였다.
    Returns: 제거된 중복 개수"""
    removed = 0
    for cr in crs:
        idxs = cr.get("evidence_review_indices", [])
        seen: set = set()
        uniq: list = []
        for i in idxs:
            if i in seen:
                removed += 1
                continue
            seen.add(i)
            uniq.append(i)
        cr["evidence_review_indices"] = uniq
    return removed


def extract_segment_crs(purpose: str, reviews: list[str]) -> list[dict]:
    """세그먼트 전체 리뷰 → CR 리스트 추출 (단일 호출 or 청크 분할)"""
    if len(reviews) <= settings.MAX_REVIEWS_PER_CHUNK:
        indexed = _format_indexed_reviews(reviews)
        crs = _single_extraction(purpose, indexed)
        _dedup_evidence_indices(crs)
        return crs

    chunks = [
        reviews[i : i + settings.MAX_REVIEWS_PER_CHUNK]
        for i in range(0, len(reviews), settings.MAX_REVIEWS_PER_CHUNK)
    ]
    all_crs: list[dict] = []
    offset = 0
    for chunk in chunks:
        indexed = _format_indexed_reviews(chunk, offset)
        crs = _single_extraction(purpose, indexed)
        all_crs.extend(crs)
        print(f"    Chunk ({offset}~{offset + len(chunk) - 1}): {len(crs)}개 CR")
        offset += len(chunk)

    if len(chunks) > 1:
        print(f"  [Merge] {len(all_crs)}개 CR → 중복 병합 중...")
        all_crs = _merge_duplicate_crs(purpose, all_crs)
        print(f"  [Merge] → {len(all_crs)}개 CR")

    removed = _dedup_evidence_indices(all_crs)
    if removed:
        print(f"  [Merge] 근거 리뷰 인덱스 중복 {removed}건 제거 (합집합 보장)")

    return all_crs


def run_phase2(tagging_df: pd.DataFrame) -> tuple[dict, dict]:
    """Phase 2: 세그먼트별 CR 추출 → (cr_results, segment_reviews) 반환"""
    cr_results: dict[str, list[dict]] = {}
    segment_reviews: dict[str, list[str]] = {}

    for purpose in PURPOSES:
        segment_df = tagging_df[
            (tagging_df["Is_Valid"] == True) &
            (tagging_df["Purpose"] == purpose) &
            (tagging_df["Product_Type"] == "laptop")
        ].reset_index(drop=True)

        reviews = segment_df["Original_Review"].tolist()
        segment_reviews[purpose] = reviews
        print(f"\n[Phase 2] {purpose}: {len(reviews)}개 리뷰 → Segment-Level CR 추출")

        crs = extract_segment_crs(purpose, reviews)

        min_support = settings.MIN_REVIEW_SUPPORT
        filtered = [
            cr for cr in crs
            if len(cr.get("evidence_review_indices", [])) >= min_support
        ]
        if len(filtered) < len(crs):
            print(f"  [{purpose}] MIN_REVIEW_SUPPORT 미달 제거: {len(crs) - len(filtered)}개")

        # 세그먼트당 CR 상한 강제 — 언급 빈도(evidence 수) 내림차순 상위 N개만 유지
        filtered.sort(
            key=lambda cr: len(cr.get("evidence_review_indices", [])), reverse=True
        )
        if len(filtered) > settings.MAX_CRS_PER_SEGMENT:
            print(
                f"  [{purpose}] 상한 초과 — mention_count 상위 "
                f"{settings.MAX_CRS_PER_SEGMENT}개 유지 "
                f"({len(filtered)} → {settings.MAX_CRS_PER_SEGMENT})"
            )
            filtered = filtered[: settings.MAX_CRS_PER_SEGMENT]

        cr_results[purpose] = filtered

        print(f"  → {purpose}: {len(filtered)}개 CR 추출 완료")

    return cr_results, segment_reviews


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Final Processing
#   Priority → ChromaDB Populate → CandidateSearch → LLMClassify
# ═════════════════════════════════════════════════════════════════════════════

class FinalProcessingState(TypedDict):
    pro_crs:            List[dict]
    cas_crs:            List[dict]
    reviews_pro:        List[str]
    reviews_cas:        List[str]
    priority_done:      bool
    chromadb_done:      bool
    sim_matrix:         List[List[float]]
    current_idx:        int
    matched_cas:        List[int]
    common_crs:         List[dict]
    specific_pro:       List[dict]
    specific_cas:       List[dict]
    _candidate_pairs:   List[dict]


def priority_node(state: FinalProcessingState) -> dict:
    """evidence_review_indices 기반 mention count → 분위수 기반 1~5점

    Phase 2에서 LLM이 전체 리뷰를 보고 각 CR에 직접 매핑한
    evidence_review_indices의 길이가 곧 해당 CR의 언급 빈도이다.
    """
    print("\n[Priority] evidence 기반 mention count 산정 중...")

    def score_crs(purpose: str, crs: List[dict]) -> List[dict]:
        if not crs:
            return crs

        for cr in crs:
            cr["mention_count"] = len(cr.get("evidence_review_indices", []))

        counts = [cr["mention_count"] for cr in crs]
        q20, q40, q60, q80 = (np.percentile(counts, p) for p in [20, 40, 60, 80])

        def to_priority(c: int) -> int:
            if c <= q20:   return 1
            elif c <= q40: return 2
            elif c <= q60: return 3
            elif c <= q80: return 4
            else:          return 5

        for cr in crs:
            cr["priority"] = to_priority(cr["mention_count"])
            print(f"  {cr['label'][:50]:<50} | count={cr['mention_count']:>4} | priority={cr['priority']}")

        return crs

    pro_scored = score_crs(settings.SEGMENT_A, state["pro_crs"])
    cas_scored = score_crs(settings.SEGMENT_B, state["cas_crs"])

    return {
        "pro_crs":       pro_scored,
        "cas_crs":       cas_scored,
        "priority_done": True,
    }


def chromadb_node(state: FinalProcessingState) -> dict:
    """Phase 2 결과를 ChromaDB에 저장 — Stage 1 호환"""
    print("\n[ChromaDB] Stage 1 호환 데이터 적재 중...")

    for purpose, crs_key, reviews_key in [
        (settings.SEGMENT_A, "pro_crs", "reviews_pro"),
        (settings.SEGMENT_B, "cas_crs", "reviews_cas"),
    ]:
        crs = state[crs_key]
        reviews = state[reviews_key]
        cr_db = cr_collections[purpose]
        review_db = review_collections[purpose]

        for cr in crs:
            label = cr.get("label", "")
            cr_db.add_texts([label], metadatas=[{"purpose": purpose}])

            for idx in cr.get("evidence_review_indices", []):
                if 0 <= idx < len(reviews):
                    review_db.add_texts(
                        [f"{reviews[idx]} → {label}"],
                        metadatas=[{"cr_label": label, "purpose": purpose}],
                    )

        print(f"  [{purpose}] {len(crs)}개 CR + evidence 리뷰 적재 완료")

    return {"chromadb_done": True}


def candidate_search_node(state: FinalProcessingState) -> dict:
    """코사인 유사도로 후보 쌍 탐색 (실제 판정은 LLM이 담당)"""
    pro_crs = state["pro_crs"]
    cas_crs = state["cas_crs"]

    print("\n[Classification] 후보 쌍 탐색 중...")

    pro_labels = [cr.get("label", "") for cr in pro_crs]
    cas_labels = [cr.get("label", "") for cr in cas_crs]

    pro_embs   = embeddings.embed_documents(pro_labels)
    cas_embs   = embeddings.embed_documents(cas_labels)
    sim_matrix = cosine_similarity(pro_embs, cas_embs)

    candidate_pairs = []
    for i, pro_cr in enumerate(pro_crs):
        for j, cas_cr in enumerate(cas_crs):
            sim = sim_matrix[i][j]
            if sim >= settings.CANDIDATE_SIM_THRESHOLD:
                candidate_pairs.append({
                    "pro_idx": i,
                    "cas_idx": j,
                    "pro_cr":  pro_cr,
                    "cas_cr":  cas_cr,
                    "sim":     round(float(sim), 3),
                })

    candidate_pairs.sort(key=lambda x: x["sim"], reverse=True)
    print(f"  후보 쌍: {len(candidate_pairs)}개 (threshold={settings.CANDIDATE_SIM_THRESHOLD})")

    return {
        "sim_matrix":       sim_matrix.tolist(),
        "current_idx":      0,
        "matched_cas":      [],
        "common_crs":       [],
        "specific_pro":     [],
        "specific_cas":     [],
        "_candidate_pairs": candidate_pairs,
    }


def _voc_texts(cr: dict, reviews: List[str]) -> List[str]:
    """CR의 evidence 인덱스를 리뷰 원문으로 해석

    인덱스는 세그먼트별 리뷰 리스트에 대한 로컬 값이므로 재실행 시 어긋날 수
    있다. 따라서 인덱스가 아니라 원문 텍스트를 산출물에 남긴다.
    """
    return [
        str(reviews[i]).strip()
        for i in cr.get("evidence_review_indices", [])
        if isinstance(i, int) and 0 <= i < len(reviews)
    ]


def llm_classify_node(state: FinalProcessingState) -> dict:
    """후보 쌍 전체를 LLM이 내용 기반으로 Common / Specific 판정"""
    pro_crs         = state["pro_crs"]
    cas_crs         = state["cas_crs"]
    reviews_pro     = state["reviews_pro"]
    reviews_cas     = state["reviews_cas"]
    candidate_pairs = state.get("_candidate_pairs", [])
    matched_pro     = set()
    matched_cas     = set()
    common_crs      = []

    print(f"  [LLM 판정] {len(candidate_pairs)}개 후보 쌍 처리 중...")

    for pair in candidate_pairs:
        i      = pair["pro_idx"]
        j      = pair["cas_idx"]
        pro_cr = pair["pro_cr"]
        cas_cr = pair["cas_cr"]

        if i in matched_pro or j in matched_cas:
            continue

        pro_label = pro_cr.get("label", "")
        cas_label = cas_cr.get("label", "")

        try:
            response = llm.invoke([
                SystemMessage(content=prompts.COMMON_JUDGE_PROMPT.format(
                    pro_cr=pro_label, cas_cr=cas_label,
                )),
                HumanMessage(content="Please judge."),
            ])
            result    = json.loads(
                response.content.replace("```json", "").replace("```", "").strip()
            )
            is_common = result.get("is_common", False)
            reasoning = result.get("reasoning", "")
            print(f"  [{'COMMON' if is_common else 'SPECIFIC'}] '{pro_label[:35]}' ↔ '{cas_label[:35]}'")
            print(f"    → {reasoning}")

            if is_common:
                canonical = result.get("canonical_label") or pro_label
                pro_pri = pro_cr.get("priority", 0) or 0
                cas_pri = cas_cr.get("priority", 0) or 0
                unified_priority = round((pro_pri + cas_pri) / 2)
                common_crs.append({
                    "canonical_label": canonical,
                    "priority":        unified_priority,
                    "similarity":      pair["sim"],
                    "reasoning":       reasoning,
                    # 근거 리뷰 원문 — Common CR은 양 세그먼트에서 병합되므로
                    # 세그먼트별로 나누어 보존한다
                    "voc_evidence": {
                        settings.SEGMENT_A: _voc_texts(pro_cr, reviews_pro),
                        settings.SEGMENT_B: _voc_texts(cas_cr, reviews_cas),
                    },
                })
                matched_pro.add(i)
                matched_cas.add(j)
        except Exception as e:
            print(f"  [Error] {e}")
            continue

    specific_pro = [cr for i, cr in enumerate(pro_crs) if i not in matched_pro]
    specific_cas = [cr for j, cr in enumerate(cas_crs) if j not in matched_cas]

    for cr in specific_pro:
        cr["voc_evidence"] = {settings.SEGMENT_A: _voc_texts(cr, reviews_pro)}
    for cr in specific_cas:
        cr["voc_evidence"] = {settings.SEGMENT_B: _voc_texts(cr, reviews_cas)}

    print(f"\n  → Common: {len(common_crs)}개")
    print(f"  → Specific Professional: {len(specific_pro)}개")
    print(f"  → Specific Casual: {len(specific_cas)}개")

    return {
        "common_crs":   common_crs,
        "specific_pro": specific_pro,
        "specific_cas": specific_cas,
        "matched_cas":  list(matched_cas),
        "current_idx":  len(pro_crs),
    }


def cr_dedup_node(state: FinalProcessingState) -> dict:
    """전체 CR(Common + Specific 양쪽)에 걸친 의미 중복을 병합

    LLMClassify 는 Pro-Cas 쌍 단위로만 판정하므로 두 종류의 중복이 남는다.
      (1) 확정된 Common CR 사이의 중복
      (2) Common CR 과 한쪽 Specific CR 사이의 중복
          — Specific 은 애초에 판정 대상이 아니었으므로 그대로 살아남는다
    따라서 검사 범위를 전체 CR 로 둔다.

    CR 이 모두 "I want/need my laptop to ..." 템플릿이라 코사인 유사도의
    기저값이 높다(무관한 쌍도 0.90 내외). 유사도는 후보 탐색에만 쓰고 판정은
    LLM 이 수행한다 — CandidateSearch → LLMClassify 와 동일한 2단 구조.

    흡수 우선순위:
      Common > Specific   (Common 은 양 세그먼트와 관련되므로 보존)
      Common 끼리 → priority 높은 쪽
      Specific 끼리 → 언급 빈도 높은 쪽
    """
    buckets = [
        ("common", state["common_crs"], "canonical_label"),
        ("pro", state["specific_pro"], "label"),
        ("cas", state["specific_cas"], "label"),
    ]
    items: list[dict] = []
    for kind, crs, key in buckets:
        for idx, cr in enumerate(crs):
            items.append({
                "kind": kind, "idx": idx, "cr": cr,
                "label": str(cr.get(key) or cr.get("requirement") or ""),
            })

    if len(items) < 2:
        return {}

    print("\n[Dedup] 전체 CR 중복 검사 중...")
    print(f"  검사 대상: Common {len(state['common_crs'])} + "
          f"{settings.SEGMENT_A} {len(state['specific_pro'])} + "
          f"{settings.SEGMENT_B} {len(state['specific_cas'])} = {len(items)}개")

    labels = [it["label"] for it in items]
    sims = cosine_similarity(embeddings.embed_documents(labels))

    candidates = sorted(
        (
            (sims[i][j], i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if sims[i][j] >= settings.INTRA_CR_DEDUP_THRESHOLD
        ),
        reverse=True,
    )
    print(f"  후보 쌍: {len(candidates)}개 (threshold={settings.INTRA_CR_DEDUP_THRESHOLD})")

    def _rank(it: dict) -> tuple[int, float]:
        """흡수 우선순위 — 큰 쪽이 살아남는다"""
        kind_rank = 1 if it["kind"] == "common" else 0
        cr = it["cr"]
        weight = (cr.get("priority", 0) or 0) if it["kind"] == "common" \
            else len(cr.get("evidence_review_indices", []))
        return (kind_rank, float(weight))

    absorbed: set[int] = set()
    for sim, i, j in candidates:
        if i in absorbed or j in absorbed:
            continue

        try:
            response = llm.invoke([
                SystemMessage(content=prompts.CR_DEDUP_JUDGE_PROMPT.format(
                    cr_a=labels[i], cr_b=labels[j],
                )),
                HumanMessage(content="Please judge."),
            ])
            result = json.loads(
                response.content.replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            print(f"  [Error] {e} — 보수적으로 유지")
            continue

        tag_i, tag_j = items[i]["kind"], items[j]["kind"]
        if not result.get("is_duplicate", False):
            print(f"  [KEEP ] ({tag_i}) '{labels[i][:32]}' ↔ ({tag_j}) '{labels[j][:32]}' "
                  f"sim={sim:.3f}")
            continue

        keep, drop = (i, j) if _rank(items[i]) >= _rank(items[j]) else (j, i)
        k_cr, d_cr = items[keep]["cr"], items[drop]["cr"]

        # 라벨은 흡수하는 쪽의 종류에 맞는 키에 기록
        merged = result.get("canonical_label") or labels[keep]
        if items[keep]["kind"] == "common":
            k_cr["canonical_label"] = merged
            k_cr["priority"] = max(k_cr.get("priority", 0) or 0,
                                   d_cr.get("priority", 0) or 0)
        else:
            k_cr["label"] = merged

        # 근거(원문 + 인덱스) 병합
        merged_voc: dict = k_cr.get("voc_evidence", {})
        for seg, texts in d_cr.get("voc_evidence", {}).items():
            kept_texts = merged_voc.setdefault(seg, [])
            kept_texts.extend(t for t in texts if t not in kept_texts)
        k_cr["voc_evidence"] = merged_voc

        if items[keep]["kind"] != "common":
            idxs = k_cr.setdefault("evidence_review_indices", [])
            idxs.extend(x for x in d_cr.get("evidence_review_indices", []) if x not in idxs)
            k_cr["mention_count"] = len(idxs)

        print(f"  [MERGE] ({tag_j if drop == j else tag_i}) '{labels[drop][:32]}' "
              f"→ ({items[keep]['kind']}) '{merged[:32]}' sim={sim:.3f}")
        print(f"    → {result.get('reasoning', '')[:110]}")
        absorbed.add(drop)

    kept = {"common": [], "pro": [], "cas": []}
    for n, it in enumerate(items):
        if n not in absorbed:
            kept[it["kind"]].append(it["cr"])

    print(f"  → 병합 결과: Common {len(state['common_crs'])}→{len(kept['common'])} · "
          f"{settings.SEGMENT_A} {len(state['specific_pro'])}→{len(kept['pro'])} · "
          f"{settings.SEGMENT_B} {len(state['specific_cas'])}→{len(kept['cas'])} "
          f"(총 {len(items)}→{len(items) - len(absorbed)}개)")

    return {
        "common_crs": kept["common"],
        "specific_pro": kept["pro"],
        "specific_cas": kept["cas"],
    }




_phase3_workflow = StateGraph(FinalProcessingState)
_phase3_workflow.add_node("Priority",        priority_node)
_phase3_workflow.add_node("ChromaDB",        chromadb_node)
_phase3_workflow.add_node("CandidateSearch", candidate_search_node)
_phase3_workflow.add_node("LLMClassify",     llm_classify_node)
_phase3_workflow.add_node("CRDedup",         cr_dedup_node)
_phase3_workflow.set_entry_point("Priority")
_phase3_workflow.add_edge("Priority",        "ChromaDB")
_phase3_workflow.add_edge("ChromaDB",        "CandidateSearch")
_phase3_workflow.add_edge("CandidateSearch", "LLMClassify")
_phase3_workflow.add_edge("LLMClassify",     "CRDedup")
_phase3_workflow.add_edge("CRDedup",         END)
phase3_app = _phase3_workflow.compile()


def run_phase3(cr_results: dict, segment_reviews: dict) -> dict:
    """Phase 3: Priority → ChromaDB → Classification → final result dict 반환"""
    pro_crs = cr_results.get(settings.SEGMENT_A, [])
    cas_crs = cr_results.get(settings.SEGMENT_B, [])

    print("[Phase 3] Final Processing 시작...")

    final_state = phase3_app.invoke({
        "pro_crs":          pro_crs,
        "cas_crs":          cas_crs,
        "reviews_pro":      segment_reviews.get(settings.SEGMENT_A, []),
        "reviews_cas":      segment_reviews.get(settings.SEGMENT_B, []),
        "priority_done":    False,
        "chromadb_done":    False,
        "sim_matrix":       [],
        "current_idx":      0,
        "matched_cas":      [],
        "common_crs":       [],
        "specific_pro":     [],
        "specific_cas":     [],
        "_candidate_pairs": [],
    })

    return {
        "common":       final_state["common_crs"],
        "specific_pro": final_state["specific_pro"],
        "specific_cas": final_state["specific_cas"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_specific_crs(crs: list[dict]) -> list[dict]:
    """Phase 2/3 내부 키 'label' → Stage 1 호환 키 'requirement' 변환"""
    out = []
    for cr in crs:
        out.append({
            "requirement": cr.get("label", cr.get("requirement", "")),
            "priority":    cr.get("priority"),
            "mention_count": cr.get("mention_count"),
            "evidence_review_indices": cr.get("evidence_review_indices", []),
            "voc_evidence": cr.get("voc_evidence", {}),
        })
    return out


def _voc_summary(cr: dict) -> tuple[int, str]:
    """(근거 리뷰 수, 미리보기 발췌) 반환

    미리보기는 열람 편의용이며 방법론적 '대표 리뷰'가 아니다. 인용은
    VoC_Evidence 시트의 전체 목록에서 선택한다.
    """
    texts = [t for seg in cr.get("voc_evidence", {}).values() for t in seg]
    if not texts:
        return 0, ""
    sample = " ".join(str(texts[0]).split())
    return len(texts), (sample[:200] + "…" if len(sample) > 200 else sample)


def _build_voc_evidence_df(result: dict) -> pd.DataFrame:
    """CR별 근거 리뷰 원문 전체를 롱 포맷으로 펼침"""
    rows: list[dict] = []
    groups = [
        ("Common", result["common"], "canonical_label"),
        (f"Specific_{settings.SEGMENT_A}", result["specific_pro"], "requirement"),
        (f"Specific_{settings.SEGMENT_B}", result["specific_cas"], "requirement"),
    ]
    for cr_type, crs, label_key in groups:
        for cr in crs:
            label = cr.get(label_key) or cr.get("label", "")
            for seg, texts in cr.get("voc_evidence", {}).items():
                for n, text in enumerate(texts, start=1):
                    rows.append({
                        "CR_Type":     cr_type,
                        "CR":          label,
                        "Segment":     seg,
                        "Evidence_No": n,
                        "Review_Text": text,
                    })
    return pd.DataFrame(rows)


def translate_crs_to_voc(result: dict) -> None:
    """확정된 CR 전체에 표시용 VoC 문장(voc 필드)을 병기.

    파이프라인 내부 CR 텍스트(키)는 절대 불변 — 산출물·매뉴얼·대시보드
    표시용 필드만 추가한다. 변환 누락 시 원문으로 대체(빈 값 금지)."""
    entries: list[tuple[dict, str]] = []
    for key, label_key in [
        ("common", "canonical_label"),
        ("specific_pro", "requirement"),
        ("specific_cas", "requirement"),
    ]:
        for cr in result.get(key, []):
            label = cr.get(label_key) or cr.get("label") or ""
            if label:
                entries.append((cr, label))
    if not entries:
        return

    listing = "\n".join(
        f'CR-{i}: "{label}"' for i, (_, label) in enumerate(entries, 1)
    )
    user_msg = (
        "Rewrite each CR below as a plain customer-voice sentence.\n\n"
        f"### CRs:\n{listing}\n\n"
        'Output JSON: {"voc": {"CR-1": "...", "CR-2": "..."}}'
    )

    print(f"\n[VoC] 표시용 고객 발화 변환 — {len(entries)}개 CR")
    resp = llm.invoke([
        {"role": "system", "content": prompts.VOC_TRANSLATION_SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    try:
        voc_map = json.loads(resp.content).get("voc", {})
    except json.JSONDecodeError:
        voc_map = {}

    missing = 0
    for i, (cr, label) in enumerate(entries, 1):
        v = str(voc_map.get(f"CR-{i}", "") or "").strip()
        if not v:
            missing += 1
            v = label
        cr["voc"] = v
    if missing:
        print(f"  ⚠️  [VoC] {missing}개 CR 변환 누락 — 원문으로 대체")
    else:
        print(f"  [VoC] {len(entries)}개 전부 변환 완료")


def _to_sheet_df(crs: list[dict], label_key: str) -> pd.DataFrame:
    """VoC 요약 컬럼을 붙이고 Excel에 담을 수 없는 중첩 필드를 제거"""
    rows = []
    for cr in crs:
        count, sample = _voc_summary(cr)
        row = {k: v for k, v in cr.items() if k != "voc_evidence"}
        row["voc_evidence_count"] = count
        row["voc_sample"] = sample
        rows.append(row)
    df = pd.DataFrame(rows)
    if label_key in df.columns:
        cols = [label_key] + [c for c in df.columns if c != label_key]
        df = df[cols]
    return df


def run() -> None:
    os.makedirs(settings.INTERMEDIATE_PATH, exist_ok=True)
    os.makedirs(settings.CHROMA_BASE_PATH,  exist_ok=True)

    # ── Phase 1 ──
    # 태깅은 CR 개수 설정과 무관하므로, 기존 결과가 있으면 재사용해 입력을 고정한다.
    # (재실행 간 비교 시 태깅 결과 변동이 교란 변수가 되는 것을 막는 목적)
    tagged_path = f"{settings.INTERMEDIATE_PATH}/tagged_reviews.xlsx"

    if settings.REUSE_TAGGED_REVIEWS and os.path.exists(tagged_path):
        tagged_df = pd.read_excel(tagged_path)
        print(f"\n[Phase 1] 건너뜀 — 기존 태깅 재사용: {tagged_path} ({len(tagged_df)}행)")
        print("  Purpose 분포:")
        print(tagged_df[tagged_df["Is_Valid"] == True]["Purpose"].value_counts())
    else:
        raw_data = pd.read_excel(settings.RAW_DATA_PATH)
        tagged_df = run_phase1(raw_data)
        # 원본 추적 컬럼 병기 (results는 위치 인덱스 순으로 정렬되어 있음)
        for col in ("Post_ID", "Date"):
            if col in raw_data.columns:
                tagged_df[col] = raw_data[col].values
        tagged_df.to_excel(tagged_path, index=False)
        print(f"저장 완료: {tagged_path}")

    # ── Phase 2 ──
    cr_results, segment_reviews = run_phase2(tagged_df)
    phase2_path = f"{settings.INTERMEDIATE_PATH}/phase2_cr_extraction.xlsx"
    with pd.ExcelWriter(phase2_path, engine="openpyxl") as writer:
        for purpose, crs in cr_results.items():
            if crs:
                pd.DataFrame(crs).to_excel(writer, sheet_name=purpose, index=False)
    print(f"저장 완료: {phase2_path}")

    # ── Phase 3 ──
    result = run_phase3(cr_results, segment_reviews)
    result["specific_pro"] = _normalize_specific_crs(result["specific_pro"])
    result["specific_cas"] = _normalize_specific_crs(result["specific_cas"])

    # 표시용 VoC 병기 (내부 CR 텍스트는 불변)
    translate_crs_to_voc(result)

    output_path = f"{settings.INTERMEDIATE_PATH}/final_cr_output.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if result["common"]:
            _to_sheet_df(result["common"], "canonical_label").to_excel(
                writer, sheet_name="Common_CR", index=False
            )
        for purpose, key in [(settings.SEGMENT_A, "specific_pro"), (settings.SEGMENT_B, "specific_cas")]:
            if result[key]:
                _to_sheet_df(result[key], "requirement").to_excel(
                    writer, sheet_name=f"Specific_{purpose}", index=False
                )

        voc_df = _build_voc_evidence_df(result)
        if not voc_df.empty:
            voc_df.to_excel(writer, sheet_name="VoC_Evidence", index=False)
    print(f"저장 완료: {output_path}")

    # 요약
    print("\n" + "=" * 50)
    print("  최종 CR 요약")
    print("=" * 50)
    print(f"  Common CR              : {len(result['common'])}개")
    print(f"  Professional (Specific): {len(result['specific_pro'])}개")
    print(f"  Casual (Specific)      : {len(result['specific_cas'])}개")
    print(f"  VoC Evidence           : {len(voc_df)}행 "
          f"(CR {voc_df['CR'].nunique() if not voc_df.empty else 0}개)")


if __name__ == "__main__":
    run()
