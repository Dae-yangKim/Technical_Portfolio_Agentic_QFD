# Stage 1: 통합 EC Debate → 3블록 Matrix Fill → 통합 HoQ 구성
#
# Input:  data/intermediate/final_cr_output.xlsx   (Stage 0 결과)
#         ChromaDB: data/chroma/chroma_db_review_{purpose}/
# Output: data/intermediate/hoq.json
#
# Phase 1: Stage 0 결과 로드 → Common / Specific Pro / Specific Cas 3블록 분리
# Phase 2: 통합 EC Debate (Phase 2-A: EC 토론 + Phase 2-B: Roof 토론)
# Phase 3: 3블록 Matrix Fill (Common 1회 + Specific Pro + Specific Cas)
# Phase 4: 결과 저장 + 검증
import json, os
import pandas as pd

import settings
from src.schemas.state import QFDState
from src.agents.debate import run_hoq_debate, fill_segment_matrix


PURPOSES = [settings.SEGMENT_A, settings.SEGMENT_B]


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Stage 0 결과 로드 → 3블록 CR 구성
# ═════════════════════════════════════════════════════════════════════════════

def load_stage0_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage 0 최종 결과 (final_cr_output.xlsx) 로드

    Returns:
        (common_df, specific_pro_df, specific_cas_df)
    """
    path = f"{settings.INTERMEDIATE_PATH}/final_cr_output.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 0 결과를 찾을 수 없습니다: {path}\n"
            "Stage 0 을 먼저 실행해주세요."
        )

    xls = pd.ExcelFile(path)
    common_df = pd.read_excel(xls, sheet_name="Common_CR")
    specific_pro_df = pd.read_excel(xls, sheet_name=f"Specific_{settings.SEGMENT_A}")
    specific_cas_df = pd.read_excel(xls, sheet_name=f"Specific_{settings.SEGMENT_B}")

    print(f"[Phase 1] Stage 0 결과 로드 완료")
    print(f"  Common CR              : {len(common_df)}개")
    print(f"  Specific {settings.SEGMENT_A:12s} : {len(specific_pro_df)}개")
    print(f"  Specific {settings.SEGMENT_B:12s} : {len(specific_cas_df)}개")

    return common_df, specific_pro_df, specific_cas_df


def _df_to_cr_list(df: pd.DataFrame, label_col: str, priority_col: str = "priority") -> list[dict]:
    """DataFrame → [{"cr": str, "priority": float, ("voc": str)}, ...] 변환

    voc = 표시용 고객 발화 병기 필드 (Stage 0 생성). 내부 키는 여전히 "cr"."""
    has_voc = "voc" in df.columns
    out = []
    for _, row in df.iterrows():
        item = {"cr": row[label_col], "priority": float(row[priority_col])}
        if has_voc and pd.notna(row["voc"]) and str(row["voc"]).strip():
            item["voc"] = str(row["voc"])
        out.append(item)
    return out


def build_cr_blocks(
    common_df: pd.DataFrame,
    specific_pro_df: pd.DataFrame,
    specific_cas_df: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Stage 0 결과를 Common / Specific Pro / Specific Cas 3블록으로 분리

    Returns:
        {
            "common":       [{"cr": str, "priority": float}, ...],
            "specific_pro": [{"cr": str, "priority": float}, ...],
            "specific_cas": [{"cr": str, "priority": float}, ...],
        }
    """
    return {
        "common":       _df_to_cr_list(common_df, "canonical_label"),
        "specific_pro": _df_to_cr_list(specific_pro_df, "requirement"),
        "specific_cas": _df_to_cr_list(specific_cas_df, "requirement"),
    }


def run_phase1() -> dict[str, list[dict]]:
    """Stage 0 결과를 로드하고 3블록 CR 구조를 반환

    Returns:
        {"common": [...], "specific_pro": [...], "specific_cas": [...]}
    """
    common_df, specific_pro_df, specific_cas_df = load_stage0_results()
    cr_blocks = build_cr_blocks(common_df, specific_pro_df, specific_cas_df)

    print(f"  Common CR      : {len(cr_blocks['common'])}개")
    print(f"  Specific Pro   : {len(cr_blocks['specific_pro'])}개")
    print(f"  Specific Cas   : {len(cr_blocks['specific_cas'])}개")
    total = sum(len(v) for v in cr_blocks.values())
    print(f"  Total unique   : {total}개")

    return cr_blocks


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: 통합 EC Debate (Phase 2-A: EC 토론 + Phase 2-B: Roof 토론)
# ═════════════════════════════════════════════════════════════════════════════

def run_ec_debate(cr_blocks: dict[str, list[dict]]) -> dict:
    """전체 CR을 고려한 통합 EC Debate 실행

    EC 토론에는 양쪽 세그먼트가 보는 전체 CR이 필요:
      cr_list_a = Common + Specific Pro
      cr_list_b = Common + Specific Cas

    Phase 2-A: EC 리스트 토론 (에이전트가 도구를 사용하여 근거 수집)
    Phase 2-B: 확정된 EC에 대해 Roof Matrix 토론

    Returns:
        dict with ec_list, roof_matrix, iteration, consensus, messages, ec_rounds, roof_rounds
    """
    cr_list_a = cr_blocks["common"] + cr_blocks["specific_pro"]
    cr_list_b = cr_blocks["common"] + cr_blocks["specific_cas"]

    initial_state: QFDState = {
        "segment_a": settings.SEGMENT_A,
        "segment_b": settings.SEGMENT_B,
        "cr_list_a": cr_list_a,
        "cr_list_b": cr_list_b,
        "iteration": 0,
        "consensus": False,
        "messages": [],
    }

    print(f"\n[Phase 2] 통합 EC Debate 시작 (MAX_ROUNDS={settings.MAX_DEBATE_ROUNDS})")
    print(f"  {settings.SEGMENT_A} 전체 CR: {len(cr_list_a)}개 (Common {len(cr_blocks['common'])} + Specific {len(cr_blocks['specific_pro'])})")
    print(f"  {settings.SEGMENT_B} 전체 CR: {len(cr_list_b)}개 (Common {len(cr_blocks['common'])} + Specific {len(cr_blocks['specific_cas'])})")

    result = run_hoq_debate(initial_state)

    print(f"\n  ✅ 통합 EC Debate 완료")
    print(f"     EC 토론   : {result.get('ec_rounds', '?')}라운드")
    print(f"     Roof 토론 : {result.get('roof_rounds', '?')}라운드")
    print(f"     공유 EC 수: {len(result['ec_list'])}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: 3블록 Matrix Fill (공유 EC × CR 블록별)
# ═════════════════════════════════════════════════════════════════════════════

def run_matrix_fill(
    ec_list: list[dict],
    roof_matrix: dict,
    cr_blocks: dict[str, list[dict]],
    debate_messages: list[dict] | None = None,
) -> dict[str, dict]:
    """Common / Specific Pro / Specific Cas 각각에 대해 Matrix Fill 수행

    Common CRs:       segment=None (양쪽 리뷰 모두 참조, 기술적 관계 평가)
    Specific Pro CRs: segment="Professional"
    Specific Cas CRs: segment="Casual"

    Returns:
        {
            "common":                {...matrices},
            "specific_professional": {...matrices},
            "specific_casual":       {...matrices},
        }
    """
    print(f"\n[Phase 3] 3블록 Matrix Fill 시작")

    results = {}

    # ① Common CRs — 1회, 세그먼트 무관 (기술적 관계 평가)
    if cr_blocks["common"]:
        common_matrices = fill_segment_matrix(ec_list, cr_blocks["common"], segment=None)
        results["common"] = {
            "cr_list": cr_blocks["common"],
            **common_matrices,
        }
        print(f"  Common        : {len(cr_blocks['common'])} CRs × {len(ec_list)} ECs")

    # ② Specific Pro CRs
    if cr_blocks["specific_pro"]:
        pro_matrices = fill_segment_matrix(ec_list, cr_blocks["specific_pro"], segment=settings.SEGMENT_A)
        results["specific_professional"] = {
            "cr_list": cr_blocks["specific_pro"],
            **pro_matrices,
        }
        print(f"  Specific Pro  : {len(cr_blocks['specific_pro'])} CRs × {len(ec_list)} ECs")

    # ③ Specific Cas CRs
    if cr_blocks["specific_cas"]:
        cas_matrices = fill_segment_matrix(ec_list, cr_blocks["specific_cas"], segment=settings.SEGMENT_B)
        results["specific_casual"] = {
            "cr_list": cr_blocks["specific_cas"],
            **cas_matrices,
        }
        print(f"  Specific Cas  : {len(cr_blocks['specific_cas'])} CRs × {len(ec_list)} ECs")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: 결과 저장 + 검증
# ═════════════════════════════════════════════════════════════════════════════

def _block_to_serializable(block: dict) -> dict:
    """블록 결과에서 JSON 직렬화 가능한 필드만 추출"""
    return {
        "cr_list": block.get("cr_list", []),
        "relationship_matrix": block.get("relationship_matrix", {}),
        "importance_matrix": block.get("importance_matrix", {}),
        "direction_matrix": block.get("direction_matrix", {}),
    }


def verify_derived_matrices(
    result: dict, label: str, ec_names: list[str] | None = None
) -> bool:
    """Importance = abs(Relationship), Direction = sign(Relationship) 검증
    + relationship 자체의 CR/EC 키가 정본과 일치하는지 검사

    Returns:
        True if all values are consistent
    """
    rel = result["relationship_matrix"]
    imp = result["importance_matrix"]
    dir_ = result["direction_matrix"]
    errors = []

    # ── 키 정합 검사 — rel의 키가 깨져 있으면 파생 검사만으로는 통과해 버림 ──
    canonical_crs = {c["cr"] for c in result.get("cr_list", [])}
    canonical_ecs = set(ec_names) if ec_names else set()
    for cr in rel:
        if canonical_crs and cr not in canonical_crs:
            errors.append(f"  KEY: 정본에 없는 CR 키 — {cr[:60]!r}")
        if canonical_ecs:
            for ec in rel[cr]:
                if ec not in canonical_ecs:
                    errors.append(f"  KEY: 정본에 없는 EC 키 — {ec[:60]!r}")

    for cr in rel:
        for ec, val in rel[cr].items():
            expected_imp = abs(val)
            expected_dir = 1 if val > 0 else (-1 if val < 0 else 0)

            if imp.get(cr, {}).get(ec) != expected_imp:
                errors.append(f"  IMP: {cr[:30]}x{ec} = {imp.get(cr, {}).get(ec)}, expected {expected_imp}")
            if dir_.get(cr, {}).get(ec) != expected_dir:
                errors.append(f"  DIR: {cr[:30]}x{ec} = {dir_.get(cr, {}).get(ec)}, expected {expected_dir}")

    if errors:
        print(f"  ❌ [{label}] {len(errors)}개 불일치:")
        for e in errors[:5]:
            print(e)
        return False
    else:
        cr_count = len(rel)
        ec_count = len(next(iter(rel.values()))) if rel else 0
        print(f"  ✅ [{label}] {cr_count} CRs x {ec_count} ECs — 모든 파생 값 정합")
        return True


def run_phase4(results: dict[str, dict], debate_result: dict | None = None) -> str:
    """통합 HoQ를 JSON으로 저장하고 검증 수행

    Args:
        results: {"common": {...}, "specific_professional": {...}, "specific_casual": {...}}
        debate_result: 통합 EC Debate 결과 (메타데이터 저장용)

    Returns:
        저장된 파일 경로
    """
    os.makedirs(settings.INTERMEDIATE_PATH, exist_ok=True)

    hoq_output: dict = {
        "ec_list": debate_result["ec_list"] if debate_result else [],
        "roof_matrix": debate_result["roof_matrix"] if debate_result else {},
    }

    for block_key in ["common", "specific_professional", "specific_casual"]:
        if block_key in results:
            hoq_output[block_key] = _block_to_serializable(results[block_key])

    # 빈 행·빈 열 사후 검출 (발송 9 #22 — 표준 HoQ 점검): 실제 관계 행렬 기준
    ec_names_all = [ec["ec"] for ec in (debate_result.get("ec_list", []) if debate_result else [])]
    empty_cr_rows: dict[str, list[str]] = {}
    used_ecs: set[str] = set()
    for block_key in ["common", "specific_professional", "specific_casual"]:
        block = hoq_output.get(block_key)
        if not block:
            continue
        rel = block.get("relationship_matrix", {})
        rows = [cr for cr, r in rel.items() if not any(v for v in r.values())]
        if rows:
            empty_cr_rows[block_key] = rows
        for r in rel.values():
            used_ecs.update(e for e, v in r.items() if v)
    unused_ecs = [e for e in ec_names_all if e not in used_ecs]
    n_empty = sum(len(v) for v in empty_cr_rows.values())
    if n_empty:
        print(f"\n  ⚠️  [빈 행] 어떤 EC와도 관계 0인 CR {n_empty}개:")
        for bk, rows in empty_cr_rows.items():
            for cr in rows:
                print(f"      - [{bk}] {cr[:70]}")
    else:
        print(f"\n  ✅ [빈 행] 없음 — 모든 CR이 EC 1개 이상과 연결")

    # 빈 열 제거 — Stage 3의 PI=0 PC 제거와 동형 (표준 HoQ: 어떤 CR도 만족시키지
    # 않는 특성은 불필요). 관계값이 전부 0이라 TI·충돌 판정에 기여하지 않으므로
    # 제거해도 결과값은 바뀌지 않는다. EC 목록·Roof·3블록 행렬에서 함께 정리한다.
    if unused_ecs:
        print(f"  ⚠️  [빈 열] 어떤 CR과도 관계 0인 EC {len(unused_ecs)}개 → 제거: "
              + "; ".join(e[:40] for e in unused_ecs))
        drop = set(unused_ecs)
        hoq_output["ec_list"] = [ec for ec in hoq_output["ec_list"] if ec["ec"] not in drop]
        roof = hoq_output.get("roof_matrix", {})
        hoq_output["roof_matrix"] = {
            a: {b: v for b, v in row.items() if b not in drop}
            for a, row in roof.items() if a not in drop
        }
        hoq_output["roof_matrix"] = {a: r for a, r in hoq_output["roof_matrix"].items() if r}
        for block_key in ["common", "specific_professional", "specific_casual"]:
            # 저장본(hoq_output)과 원본(results) 양쪽에서 열 제거 — 이후 검증이 원본을 봄
            for block in (hoq_output.get(block_key), results.get(block_key)):
                if not block:
                    continue
                for mkey in ("relationship_matrix", "importance_matrix", "direction_matrix"):
                    if mkey in block and isinstance(block[mkey], dict):
                        block[mkey] = {
                            cr: {e: v for e, v in row.items() if e not in drop}
                            for cr, row in block[mkey].items()
                        }
        if debate_result:
            debate_result["ec_list"] = hoq_output["ec_list"]
    else:
        print(f"  ✅ [빈 열] 없음 — 모든 EC가 CR 1개 이상과 연결")

    if debate_result:
        hoq_output["_meta"] = {
            "ec_debate_rounds": debate_result.get("ec_rounds"),
            "roof_debate_rounds": debate_result.get("roof_rounds"),
            "total_rounds": debate_result.get("iteration"),
            "consensus": debate_result.get("consensus"),
            "ec_consensus": debate_result.get("ec_consensus"),
            "roof_consensus": debate_result.get("roof_consensus"),
            "shared_ec_count": len(debate_result.get("ec_list", [])),
            # 발송 9 #22 빈 행·빈 열 검사 기록
            "coverage": {
                "ec_debate_uncovered_crs": debate_result.get("ec_coverage_gaps", []),
                "empty_cr_rows": empty_cr_rows,
                "unused_ecs_removed": unused_ecs,
            },
        }

    hoq_path = f"{settings.INTERMEDIATE_PATH}/hoq.json"
    with open(hoq_path, "w", encoding="utf-8") as f:
        json.dump(hoq_output, f, ensure_ascii=False, indent=2)

    print(f"\n[Phase 4] 저장 완료: {hoq_path}")

    # 파생 매트릭스 검증 (EC 정본 목록을 함께 넘겨 키 정합까지 검사)
    ec_names = [
        ec["ec"] for ec in (debate_result.get("ec_list", []) if debate_result else [])
    ]
    for block_key, label in [
        ("common", "Common"),
        ("specific_professional", "Specific Pro"),
        ("specific_casual", "Specific Cas"),
    ]:
        if block_key in results:
            verify_derived_matrices(results[block_key], label, ec_names or None)

    return hoq_path


# ═════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인
# ═════════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Stage 1 전체 파이프라인: Phase 1 → Phase 2 → Phase 3 → Phase 4"""
    print("=" * 60)
    print("  Stage 1: HoQ Construction (Integrated HoQ)")
    print("=" * 60)

    # Phase 1: CR 3블록 로드
    cr_blocks = run_phase1()

    # Phase 2: 통합 EC Debate (2-A: EC 토론 + 2-B: Roof 토론)
    debate_result = run_ec_debate(cr_blocks)

    # Phase 3: 3블록 Matrix Fill
    results = run_matrix_fill(
        ec_list=debate_result["ec_list"],
        roof_matrix=debate_result["roof_matrix"],
        cr_blocks=cr_blocks,
        debate_messages=debate_result.get("messages", []),
    )

    # Phase 4: 저장 + 검증
    hoq_path = run_phase4(results, debate_result)

    # 요약
    print(f"\n{'=' * 60}")
    print("  Stage 1 완료 요약")
    print("=" * 60)
    shared_ec = len(debate_result["ec_list"])
    print(f"  공유 EC       : {shared_ec}개")
    print(f"  EC 토론 라운드 : {debate_result.get('ec_rounds', '?')}")
    print(f"  Roof 토론 라운드: {debate_result.get('roof_rounds', '?')}")
    for block_key, label in [("common", "Common"), ("specific_professional", "Specific Pro"), ("specific_casual", "Specific Cas")]:
        if block_key in results:
            cr_count = len(results[block_key]["relationship_matrix"])
            print(f"  {label:12s}: {cr_count} CRs x {shared_ec} ECs")
    print(f"  Output: {hoq_path}")


if __name__ == "__main__":
    run()
