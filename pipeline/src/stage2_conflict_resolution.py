"""Stage 2: 세그먼트 간 충돌 CR을 찾아 에이전트 토론과 Judge 판정으로 해결한다.

입력: data/intermediate/hoq.json
출력: data/intermediate/conflict_log.json

Phase 1: hoq.json 로드 (Importance · Direction · Roof)
Phase 2: Direction + Roof → N×N 충돌 행렬 (직접 · 간접 충돌)
Phase 3: EC 기준으로 충돌 셀을 그룹화
Phase 4: 그룹별 세그먼트 Advocate 토론 (Orchestrator 진행)
Phase 5: Judge (Market ↔ Engineering) → 전략 결정 + Importance 조정
Phase 6: 조정 반영 Importance 행렬 생성 + 저장
"""
import json, os
from collections import defaultdict

import settings
from src.schemas.conflict import ConflictCell, ConflictGroup, ResolutionLog
from src.agents.debate import run_conflict_resolution


# ── Phase 1: hoq.json 로드 ────────────────────────────────────────────────────

def load_hoq() -> dict:
    """Stage 1 결과 (hoq.json) 로드"""
    path = f"{settings.INTERMEDIATE_PATH}/hoq.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 1 결과를 찾을 수 없습니다: {path}\n"
            "Stage 1 을 먼저 실행해주세요."
        )
    with open(path, encoding="utf-8") as f:
        hoq = json.load(f)

    print(f"[Phase 1] hoq.json 로드 완료")
    print(f"  EC 수          : {len(hoq.get('ec_list', []))}개")
    for block_key, label in [
        ("common", "Common"),
        ("specific_professional", "Specific Pro"),
        ("specific_casual", "Specific Cas"),
    ]:
        block = hoq.get(block_key, {})
        cr_count = len(block.get("cr_list", []))
        print(f"  {label:14s}: {cr_count} CRs")

    return hoq


# ── Phase 2: N×N Conflict Matrix 구축 ─────────────────────────────────────────

def build_conflict_matrix(
    hoq: dict,
) -> tuple[list[ConflictCell], list[str], dict]:
    """Direction Matrix + Roof Matrix → N×N Conflict Matrix 구축

    N×N Matrix는 전체 CR(Common + Specific Pro + Specific Cas)을 행·열로 하는 정방 행렬.
    충돌 판정 전제: G ≠ H 인 CR 쌍에 대해서만 수행 (A×B, A×AB, B×AB)
    - 대각선: "-" (자기 자신)
    - 동일 세그먼트: "-" (충돌 검사 대상 아님)
    - 다른 세그먼트: {"direct": [...], "roof": [...]} 또는 None(충돌 없음)

    Returns:
        (cells, all_crs, nxn_matrix)
        - cells: 발견된 모든 충돌 인스턴스
        - all_crs: 전체 CR 순서 리스트 [Common..., Pro..., Cas...]
        - nxn_matrix: {CR_i: {CR_j: conflict_info}} 정방 행렬
    """
    common_block = hoq.get("common", {})
    pro_block = hoq.get("specific_professional", {})
    cas_block = hoq.get("specific_casual", {})
    dir_common = common_block.get("direction_matrix", {})
    dir_pro = pro_block.get("direction_matrix", {})
    dir_cas = cas_block.get("direction_matrix", {})
    roof = hoq.get("roof_matrix", {})
    ec_list = [ec["ec"] for ec in hoq.get("ec_list", [])]

    common_crs = [cr["cr"] for cr in common_block.get("cr_list", [])]
    pro_crs = [cr["cr"] for cr in pro_block.get("cr_list", [])]
    cas_crs = [cr["cr"] for cr in cas_block.get("cr_list", [])]
    all_crs = common_crs + pro_crs + cas_crs
    common_set = set(common_crs)
    pro_set = set(pro_crs)

    def _seg(cr: str) -> str:
        if cr in common_set:
            return "common"
        if cr in pro_set:
            return "pro"
        return "cas"

    cells: list[ConflictCell] = []

    # G ≠ H 인 모든 교차 세그먼트 쌍: A×B, A×AB, B×AB
    cross_pairs = [
        (dir_pro, dir_cas),
        (dir_pro, dir_common),
        (dir_cas, dir_common),
    ]

    for dir_g, dir_h in cross_pairs:
        # ── 직접 충돌: D(CR_G,i, EC_k) × D(CR_H,j, EC_k) < 0 ──
        for cr_a, dirs_a in dir_g.items():
            for cr_b, dirs_b in dir_h.items():
                for ec in ec_list:
                    d_a = dirs_a.get(ec, 0)
                    d_b = dirs_b.get(ec, 0)
                    if d_a * d_b < 0:
                        cells.append(ConflictCell(
                            cr_a=cr_a,
                            cr_b=cr_b,
                            conflict_type="direct",
                            ec=ec,
                        ))

        # ── Roof 충돌 (일반화) ──
        # CR_A의 소원 방향(d_a_k)으로 EC_k를 움직이면, 강상관을 타고 EC_l이
        # d_a_k × sign(cor) 방향으로 움직인다. 그 유발 방향이 CR_B의 소원과
        # 반대이면 간접 충돌: d_a_k × d_b_l × sign(cor) < 0.
        # (직접 충돌 d_a × d_b < 0 은 이 식의 k=l, cor=+1 특수 경우)
        # 강도 문턱은 |cor| > |τ_R| (±0.9 강상관만 간접 충돌 후보)
        tau_mag = abs(settings.ROOF_THRESHOLD)
        for cr_a, dirs_a in dir_g.items():
            for cr_b, dirs_b in dir_h.items():
                for ec_k in ec_list:
                    d_a_k = dirs_a.get(ec_k, 0)
                    if d_a_k == 0:
                        continue
                    for ec_l in ec_list:
                        if ec_k == ec_l:
                            continue
                        d_b_l = dirs_b.get(ec_l, 0)
                        if d_b_l == 0:
                            continue
                        cor = roof.get(ec_k, {}).get(ec_l, None)
                        if cor is None:
                            cor = roof.get(ec_l, {}).get(ec_k, None)
                        if cor is None or abs(cor) <= tau_mag:
                            continue
                        cor_sign = 1 if cor > 0 else -1
                        if d_a_k * d_b_l * cor_sign < 0:
                            cells.append(ConflictCell(
                                cr_a=cr_a,
                                cr_b=cr_b,
                                conflict_type="roof",
                                ec_a=ec_k,
                                ec_b=ec_l,
                            ))

    # 중복 제거 (동일 CR pair에서 동일 EC/EC pair 충돌)
    seen = set()
    unique_cells: list[ConflictCell] = []
    for cell in cells:
        key = (
            cell.cr_a, cell.cr_b, cell.conflict_type,
            cell.ec, cell.ec_a, cell.ec_b,
        )
        if key not in seen:
            seen.add(key)
            unique_cells.append(cell)

    # ── N×N 정방 행렬 구축 ──
    pair_conflicts: dict[tuple[str, str], dict] = {}
    for cell in unique_cells:
        key = (cell.cr_a, cell.cr_b)
        if key not in pair_conflicts:
            pair_conflicts[key] = {"direct": [], "roof": []}
        if cell.conflict_type == "direct":
            pair_conflicts[key]["direct"].append(cell.ec)
        else:
            pair_conflicts[key]["roof"].append(
                {"ec_a": cell.ec_a, "ec_b": cell.ec_b}
            )

    nxn_matrix: dict[str, dict] = {}
    for cr_i in all_crs:
        nxn_matrix[cr_i] = {}
        seg_i = _seg(cr_i)
        for cr_j in all_crs:
            if cr_i == cr_j:
                nxn_matrix[cr_i][cr_j] = "-"
            elif seg_i == _seg(cr_j):
                nxn_matrix[cr_i][cr_j] = "-"
            else:
                conflict = (
                    pair_conflicts.get((cr_i, cr_j))
                    or pair_conflicts.get((cr_j, cr_i))
                )
                nxn_matrix[cr_i][cr_j] = conflict

    direct_count = sum(1 for c in unique_cells if c.conflict_type == "direct")
    roof_count = sum(1 for c in unique_cells if c.conflict_type == "roof")
    print(f"\n[Phase 2] N×N Conflict Matrix 구축 완료")
    print(f"  행렬 크기     : {len(all_crs)} × {len(all_crs)}"
          f" (Common {len(common_crs)} + Pro {len(pro_crs)} + Cas {len(cas_crs)})")
    print(f"  직접 충돌     : {direct_count}개")
    print(f"  Roof 충돌     : {roof_count}개")
    print(f"  총 충돌 셀    : {len(unique_cells)}개")
    print(f"  충돌 CR pair  : {len(pair_conflicts)}개")

    return unique_cells, all_crs, nxn_matrix


# ── Phase 3: CR Classification — EC 기준 그룹화 ────────────────

def _group_cells_by_ec(
    cells: list[ConflictCell],
    all_crs: list[str],
    common_set: set[str],
    pro_set: set[str],
    ec_order: list[str],
) -> tuple[list[ConflictGroup], list[str], dict]:
    """EC 기준 충돌 그룹 생성

    1. 충돌 셀을 "무엇을 두고 다투는가"(EC)로 모은다.
       direct 셀은 그 EC 하나, roof 셀은 ec_a·ec_b 두 EC에 속한다.
    2. 두 EC는 다음 중 하나면 같은 그룹이다.
       (a) 같은 CR 쌍이 두 EC에서 모두 다툰다.
       (b) 한 roof 셀이 두 EC를 잇는다 (간접 충돌).
    3. 그룹의 CR 명단은 그 그룹 셀에 등장하는 CR 전부(세그먼트별로 나눠 기록).
       셀 1개짜리 그룹도 합치지 않고 그대로 둔다.

    Returns:
        (conflict_groups, independent_crs, grouping_meta)
    """
    meta: dict = {
        "method": "ec_based",
        "rule": "cells grouped by contested EC; ECs merged when the same CR pair "
                "conflicts on both or a roof cell links them; singleton groups kept",
        "groups": [],
    }
    if not cells:
        return [], list(all_crs), meta

    ec_index = {ec: k for k, ec in enumerate(ec_order)}

    # Union-Find over ECs
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_ecs: dict[frozenset, set[str]] = defaultdict(set)
    for cell in cells:
        pair = frozenset((cell.cr_a, cell.cr_b))
        if cell.conflict_type == "direct":
            find(cell.ec)
            pair_ecs[pair].add(cell.ec)
        else:
            union(cell.ec_a, cell.ec_b)          # (b) roof 셀이 두 EC를 잇는다
            pair_ecs[pair].update((cell.ec_a, cell.ec_b))
    for ecs in pair_ecs.values():                # (a) 같은 CR 쌍이 두 EC에서 다툰다
        ordered = sorted(ecs, key=lambda e: ec_index.get(e, 10**6))
        for e in ordered[1:]:
            union(ordered[0], e)

    comps: dict[str, dict] = defaultdict(lambda: {"ecs": set(), "cells": []})
    for cell in cells:
        root = find(cell.ec if cell.conflict_type == "direct" else cell.ec_a)
        comps[root]["cells"].append(cell)
        if cell.conflict_type == "direct":
            comps[root]["ecs"].add(cell.ec)
        else:
            comps[root]["ecs"].update((cell.ec_a, cell.ec_b))

    def _sort_key(comp: dict) -> tuple:
        return (-len(comp["cells"]),
                min(ec_index.get(e, 10**6) for e in comp["ecs"]))

    groups: list[ConflictGroup] = []
    for comp in sorted(comps.values(), key=_sort_key):
        crs = {cr for c in comp["cells"] for cr in (c.cr_a, c.cr_b)}
        ecs_sorted = sorted(comp["ecs"], key=lambda e: ec_index.get(e, 10**6))
        groups.append(ConflictGroup(
            cr_a_list=sorted(crs & pro_set),
            cr_b_list=sorted(crs - pro_set - common_set),
            cr_common_list=sorted(crs & common_set),
            cells=comp["cells"],
            ec_list=ecs_sorted,
        ))
        meta["groups"].append({
            "ecs": ecs_sorted, "cells": len(comp["cells"]), "crs": len(crs),
        })
    meta["singleton_groups"] = sum(1 for g in groups if len(g.cells) == 1)

    conflict_crs: set[str] = set()
    for g in groups:
        conflict_crs.update(g.cr_a_list, g.cr_b_list, g.cr_common_list)
    independent_crs = [cr for cr in all_crs if cr not in conflict_crs]
    return groups, independent_crs, meta


def classify_crs(
    cells: list[ConflictCell],
    hoq: dict,
) -> tuple[list[ConflictGroup], list[str], dict]:
    """CR Classification: Conflict CR 그룹(EC 기준)과 Independent CRs 분류

    Returns:
        (conflict_groups, independent_crs, grouping_meta)
    """
    common_crs = [
        cr["cr"] for cr in hoq.get("common", {}).get("cr_list", [])
    ]
    pro_crs = [
        cr["cr"] for cr in
        hoq.get("specific_professional", {}).get("cr_list", [])
    ]
    cas_crs = [
        cr["cr"] for cr in
        hoq.get("specific_casual", {}).get("cr_list", [])
    ]
    all_crs = common_crs + pro_crs + cas_crs
    ec_order = [ec["ec"] for ec in hoq.get("ec_list", [])]

    groups, independent_crs, meta = _group_cells_by_ec(
        cells, all_crs, set(common_crs), set(pro_crs), ec_order
    )

    print(f"\n[Phase 3] CR Classification 완료 (EC 기준 그룹화)")
    print(f"  Conflict Groups  : {len(groups)}개")
    for i, g in enumerate(groups):
        parts = []
        if g.cr_a_list:
            parts.append(f"Pro {len(g.cr_a_list)}")
        if g.cr_b_list:
            parts.append(f"Cas {len(g.cr_b_list)}")
        if g.cr_common_list:
            parts.append(f"Common {len(g.cr_common_list)}")
        print(f"    Group {i+1}: {' × '.join(parts)}"
              f" ({len(g.cells)} 충돌 셀) — EC: "
              + ", ".join(e[:28] for e in g.ec_list))
    if meta.get("singleton_groups"):
        print(f"  (셀 1개짜리 그룹 {meta['singleton_groups']}개 — 합치지 않고 유지)")
    print(f"  Independent CRs  : {len(independent_crs)}개")

    return groups, independent_crs, meta


# ── Phase 4 + 5: Agent-based Debate + Judge (그룹별 실행) ────────────────────────

def resolve_conflicts(
    groups: list[ConflictGroup],
    hoq: dict,
    nxn_matrix: dict,
) -> list[ResolutionLog]:
    """각 충돌 그룹에 대해 Debate → Judge 실행

    Args:
        nxn_matrix: build_conflict_matrix()에서 반환된 N×N 정방 행렬

    Returns:
        list[ResolutionLog] — 그룹별 해결 결과
    """
    if not groups:
        print("\n[Phase 4-5] 충돌 그룹 없음 — 스킵")
        return []

    resolution_logs: list[ResolutionLog] = []

    for i, group in enumerate(groups):
        print(f"\n{'='*60}")
        print(f"  Conflict Group {i+1}/{len(groups)}")
        print(f"  Pro CRs: {group.cr_a_list}")
        print(f"  Cas CRs: {group.cr_b_list}")
        if group.cr_common_list:
            print(f"  Common CRs: {group.cr_common_list}")
        print(f"{'='*60}")

        group_dict = group.model_dump()
        group_dict["cells"] = [c.model_dump() for c in group.cells]

        result = run_conflict_resolution(
            conflict_group=group_dict,
            hoq_data=hoq,
            conflict_matrix=nxn_matrix,
            group_index=i + 1,
        )

        log = ResolutionLog(
            group_cr_a=group.cr_a_list,
            group_cr_b=group.cr_b_list,
            group_cr_common=group.cr_common_list,
            strategy=result["strategy"],
            selected_segment=result.get("selected_segment"),
            reasoning=result["reasoning"],
            debate_summary=result["debate_summary"],
            judge_discussion_log=json.dumps(
                result["judge_discussion_log"],
                ensure_ascii=False,
            ),
            adjusted_importance=result["adjusted_importance"],
        )
        resolution_logs.append(log)

        strategy_label = result["strategy"]
        if strategy_label == "Selection" and result.get("selected_segment"):
            strategy_label += f" ({result['selected_segment']})"
        print(f"\n  ✅ Group {i+1} 해결 완료")
        print(f"     전략: {strategy_label}")
        print(f"     Debate: {result.get('debate_rounds', '?')}라운드")
        print(f"     Judge : {result.get('judge_rounds', '?')}라운드")

    return resolution_logs


# ── Phase 6: 결과 저장 ──────────────────────────────────────────────────────────

def build_adjusted_importance_matrix(
    hoq: dict,
    resolution_logs: list[ResolutionLog],
) -> tuple[dict[str, dict[str, int]], list[dict]]:
    """원본 Importance Matrix에 Judge 조정값을 반영한 최종 행렬 생성

    두 그룹이 같은 셀에 서로 다른 값을 내면 경고하고 기록한다
    (EC 기준 그룹화 + Judge 범위 제한으로 원칙상 발생하지 않는 안전장치).

    Returns:
        ({CR: {EC: importance}} — 전체 (Common + Specific) 통합, 충돌 기록 목록)
    """
    full_importance: dict[str, dict[str, int]] = {}
    for block_key in ["common", "specific_professional", "specific_casual"]:
        block = hoq.get(block_key, {})
        for cr, ecs in block.get("importance_matrix", {}).items():
            full_importance[cr] = dict(ecs)

    # Judge 조정값 덮어쓰기 — 원본에 없는 CR 키는 등재하지 않고 경고 후 폐기
    # (유령 키가 행렬에 정식 등재되어 하류로 전파되는 것을 차단)
    setter: dict[tuple[str, str], tuple[int, int]] = {}
    merge_conflicts: list[dict] = []
    for gi, log in enumerate(resolution_logs, 1):
        for cr, ec_vals in log.adjusted_importance.items():
            if cr in full_importance:
                for ec, val in ec_vals.items():
                    prev = setter.get((cr, ec))
                    if prev is not None and prev[1] != val:
                        merge_conflicts.append({
                            "cr": cr, "ec": ec,
                            "group_first": prev[0], "value_first": prev[1],
                            "group_last": gi, "value_last": val,
                        })
                        print(f"  ⚠️  [Adjusted Matrix] 같은 셀에 서로 다른 조정값 — "
                              f"G{prev[0]}={prev[1]} vs G{gi}={val}: "
                              f"{cr[:40]!r} × {ec[:30]!r} (뒤 그룹 값 적용)")
                    setter[(cr, ec)] = (gi, val)
                    full_importance[cr][ec] = val
            else:
                print(f"  ⚠️  [Adjusted Matrix] 원본에 없는 CR 키 폐기: {cr[:70]!r}")

    return full_importance, merge_conflicts


def save_results(
    conflict_cells: list[ConflictCell],
    all_crs: list[str],
    nxn_matrix: dict,
    groups: list[ConflictGroup],
    independent_crs: list[str],
    resolution_logs: list[ResolutionLog],
    adjusted_importance: dict,
    grouping: dict | None = None,
    merge_conflicts: list[dict] | None = None,
) -> str:
    """conflict_log.json 저장

    Returns:
        저장된 파일 경로
    """
    os.makedirs(settings.INTERMEDIATE_PATH, exist_ok=True)

    output = {
        "conflict_matrix": {
            "all_crs": all_crs,
            "nxn_matrix": nxn_matrix,
            "total_conflict_cells": len(conflict_cells),
            "direct_count": sum(
                1 for c in conflict_cells if c.conflict_type == "direct"
            ),
            "roof_count": sum(
                1 for c in conflict_cells if c.conflict_type == "roof"
            ),
            "cells": [c.model_dump() for c in conflict_cells],
        },
        "classification": {
            "conflict_groups": [g.model_dump() for g in groups],
            "independent_crs": independent_crs,
            "grouping": grouping or {"method": "ec_based"},
        },
        "resolution_log": [log.model_dump() for log in resolution_logs],
        "adjusted_importance_matrix": adjusted_importance,
        "adjustment_merge_conflicts": merge_conflicts or [],
    }

    path = f"{settings.INTERMEDIATE_PATH}/conflict_log.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[Phase 6] 저장 완료: {path}")
    return path


# ── 전체 파이프라인 ────────────────────────────────────────────────────────────────

def run() -> None:
    """Stage 2 전체 파이프라인"""
    print("=" * 60)
    print("  Stage 2: Agent-driven Conflict Resolution")
    print("=" * 60)

    hoq = load_hoq()

    conflict_cells, all_crs, nxn_matrix = build_conflict_matrix(hoq)

    groups, independent_crs, grouping_meta = classify_crs(conflict_cells, hoq)

    resolution_logs = resolve_conflicts(groups, hoq, nxn_matrix)

    adjusted_importance, merge_conflicts = build_adjusted_importance_matrix(
        hoq, resolution_logs
    )

    path = save_results(
        conflict_cells, all_crs, nxn_matrix,
        groups, independent_crs,
        resolution_logs, adjusted_importance,
        grouping=grouping_meta, merge_conflicts=merge_conflicts,
    )

    print(f"\n{'='*60}")
    print("  Stage 2 완료 요약")
    print("=" * 60)
    print(f"  총 충돌 셀       : {len(conflict_cells)}개")
    print(f"  충돌 그룹        : {len(groups)}개")
    print(f"  Independent CRs  : {len(independent_crs)}개")
    for i, log in enumerate(resolution_logs):
        print(f"  Group {i+1} 전략     : {log.strategy}")
    print(f"  Output: {path}")


if __name__ == "__main__":
    run()
