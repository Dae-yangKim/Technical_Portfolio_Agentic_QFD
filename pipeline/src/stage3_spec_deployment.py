# Stage 3: Output Synthesis + Part Deployment + PC Specification
#
# Phase 1 (3-2-3): Integrated HoQ 생성 + Technical Importance 산출
# Phase 2 (3-2-4): PC 도출 → EC-PC Matrix → PC Roof → Part Importance → Target Value
#   Separation이면: 분할 전 전체 HoQ로 공통 PC 골격 1회 → 모델별로 골격 특화(모델 맥락 근거의 추가·제외)
#   → Target Value에 모델 맥락 주입
# Phase 3 (3-2-5): PC Specification (부품 특성 사양서) — LLM 기반 설정 근거 생성
#
# Input:  data/intermediate/hoq.json, data/intermediate/conflict_log.json
# Output: data/output/integrated_hoq.json, data/output/part_deployment.json, data/output/pc_specification.xlsx
import json, os, re

import pandas as pd

import settings
from src.schemas.spec import (
    IntegratedHoQ,
    OutputSynthesis,
    PartCharacteristic,
    PartDeploymentResult,
    PartDeployment,
    PCSpecRow,
    PCSpecification,
)
from src.agents.debate import (
    run_part_deployment,
    run_pc_skeleton_debate,
    generate_pc_specification,
)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Output Synthesis — Integrated HoQ + Technical Importance
# ═════════════════════════════════════════════════════════════════════════════

def load_inputs() -> tuple[dict, dict]:
    """hoq.json + conflict_log.json 로드"""
    hoq_path = f"{settings.INTERMEDIATE_PATH}/hoq.json"
    conflict_path = f"{settings.INTERMEDIATE_PATH}/conflict_log.json"

    with open(hoq_path, encoding="utf-8") as f:
        hoq = json.load(f)
    with open(conflict_path, encoding="utf-8") as f:
        conflict_log = json.load(f)

    print(f"[Phase 1] 입력 로드 완료")
    print(f"  hoq.json: EC {len(hoq.get('ec_list', []))}개")
    print(f"  conflict_log.json: {len(conflict_log.get('resolution_log', []))} 그룹 해결됨")

    return hoq, conflict_log


def _collect_cr_list(
    hoq: dict,
    conflict_log: dict,
    importance_matrix: dict[str, dict[str, int]],
) -> list[dict]:
    """통합 CR 리스트 구성 — priority, segment, category 포함"""
    cr_list: list[dict] = []
    seen: set[str] = set()

    independent_crs = set(
        conflict_log.get("classification", {}).get("independent_crs", [])
    )

    block_segment_map = [
        ("common", "Common"),
        ("specific_professional", settings.SEGMENT_A),
        ("specific_casual", settings.SEGMENT_B),
    ]

    for block_key, segment in block_segment_map:
        for cr in hoq.get(block_key, {}).get("cr_list", []):
            cr_text = cr["cr"]
            if cr_text in seen:
                continue
            seen.add(cr_text)

            category = "independent" if cr_text in independent_crs else "conflict"
            entry = {
                "cr": cr_text,
                "priority": cr.get("priority", 1.0),
                "segment": segment,
                "category": category,
            }
            if cr.get("voc"):
                entry["voc"] = cr["voc"]
            cr_list.append(entry)

    return cr_list


def _build_full_importance_matrix(
    hoq: dict,
    conflict_log: dict,
) -> dict[str, dict[str, int]]:
    """Common + Independent (원본) + Conflict (adjusted) 통합 Importance Matrix"""
    full: dict[str, dict[str, int]] = {}

    # 원본 importance 전체 수집
    for block_key in ["common", "specific_professional", "specific_casual"]:
        block = hoq.get(block_key, {})
        for cr, ecs in block.get("importance_matrix", {}).items():
            full[cr] = dict(ecs)

    # Adjusted importance 덮어쓰기 (conflict CRs)
    adjusted = conflict_log.get("adjusted_importance_matrix", {})
    for cr, ecs in adjusted.items():
        full[cr] = dict(ecs)

    return full


def _build_full_direction_matrix(hoq: dict) -> dict[str, dict[str, int]]:
    """전체 Direction Matrix 통합"""
    full: dict[str, dict[str, int]] = {}
    for block_key in ["common", "specific_professional", "specific_casual"]:
        block = hoq.get(block_key, {})
        for cr, ecs in block.get("direction_matrix", {}).items():
            full[cr] = dict(ecs)
    return full


def compute_technical_importance(
    cr_list: list[dict],
    importance_matrix: dict[str, dict[str, int]],
    ec_list: list[dict],
) -> dict[str, float]:
    """TI(EC) = Σ(CR_priority × importance(CR, EC))"""
    ti: dict[str, float] = {}
    for ec in ec_list:
        ec_name = ec["ec"]
        total = 0.0
        for cr_info in cr_list:
            cr_text = cr_info["cr"]
            priority = cr_info["priority"]
            imp = importance_matrix.get(cr_text, {}).get(ec_name, 0)
            total += priority * imp
        ti[ec_name] = round(total, 2)
    return ti


def build_integrated_hoqs(
    hoq: dict,
    conflict_log: dict,
) -> OutputSynthesis:
    """Strategy에 따라 1개 또는 2개의 Integrated HoQ 생성"""
    ec_list = hoq.get("ec_list", [])
    roof_matrix = hoq.get("roof_matrix", {})
    resolution_log = conflict_log.get("resolution_log", [])

    # Strategy 결정 (여러 그룹이면 가장 강한 전략 채택)
    strategies = [r.get("strategy", "Compromise") for r in resolution_log]
    if "Separation" in strategies:
        strategy = "Separation"
    elif "Selection" in strategies:
        strategy = "Selection"
    else:
        strategy = "Compromise"

    selected_segment = None
    if strategy == "Selection":
        for r in resolution_log:
            if r.get("selected_segment"):
                selected_segment = r["selected_segment"]
                break

    direction_matrix = _build_full_direction_matrix(hoq)

    if strategy == "Separation":
        # ── CR 분할 방식: Separation 그룹의 충돌 CR을 세그먼트 소속대로 분배 ──
        # professional_model은 해당 그룹의 Cas CR을, casual_model은 Pro CR을
        # 보유하지 않는다. Common CR은 양쪽 모두 유지.
        excl_for_pro: set = set()  # pro 모델에서 제외 (= Separation 그룹의 Cas CR)
        excl_for_cas: set = set()  # cas 모델에서 제외 (= Separation 그룹의 Pro CR)
        for r in resolution_log:
            if r.get("strategy") == "Separation":
                excl_for_pro |= set(r.get("group_cr_b", []))
                excl_for_cas |= set(r.get("group_cr_a", []))

        base_imp = _build_full_importance_matrix(hoq, conflict_log)

        def _model_hoq(label: str, excluded: set) -> IntegratedHoQ:
            imp_m = {cr: dict(ecs) for cr, ecs in base_imp.items()
                     if cr not in excluded}
            dir_m = {cr: dict(ecs) for cr, ecs in direction_matrix.items()
                     if cr not in excluded}
            cr_list_m = [c for c in _collect_cr_list(hoq, conflict_log, imp_m)
                         if c["cr"] not in excluded]
            ti_m = compute_technical_importance(cr_list_m, imp_m, ec_list)
            if excluded:
                print(f"  [{label}] Separation CR 분할 — 제외 {len(excluded)}개:")
                for cr in sorted(excluded):
                    print(f"    - {cr[:75]}")
            return IntegratedHoQ(
                label=label,
                cr_list=cr_list_m,
                ec_list=ec_list,
                importance_matrix=imp_m,
                direction_matrix=dir_m,
                roof_matrix=roof_matrix,
                technical_importance=ti_m,
            )

        integrated_hoqs = [
            _model_hoq("professional_model", excl_for_pro),
            _model_hoq("casual_model", excl_for_cas),
        ]
    else:
        # ── 1개 HoQ 생성 ──
        importance_matrix = _build_full_importance_matrix(hoq, conflict_log)
        cr_list = _collect_cr_list(hoq, conflict_log, importance_matrix)
        ti = compute_technical_importance(cr_list, importance_matrix, ec_list)

        integrated_hoq = IntegratedHoQ(
            label="integrated",
            cr_list=cr_list,
            ec_list=ec_list,
            importance_matrix=importance_matrix,
            direction_matrix=direction_matrix,
            roof_matrix=roof_matrix,
            technical_importance=ti,
        )
        integrated_hoqs = [integrated_hoq]

    output = OutputSynthesis(
        strategy=strategy,
        selected_segment=selected_segment,
        integrated_hoqs=integrated_hoqs,
    )

    # 출력
    print(f"\n[Phase 1] Output Synthesis 완료")
    print(f"  Strategy: {strategy}")
    print(f"  Integrated HoQ 수: {len(integrated_hoqs)}")
    for h in integrated_hoqs:
        print(f"    [{h.label}] CRs: {len(h.cr_list)}, TI top 5:")
        sorted_ti = sorted(h.technical_importance.items(), key=lambda x: x[1], reverse=True)
        for rank, (ec, val) in enumerate(sorted_ti[:5], 1):
            print(f"      {rank}. {ec}: {val:.1f}")

    return output


def build_model_context(model_hoq: IntegratedHoQ, conflict_log: dict) -> str:
    """Separation 모델 맥락 — Part Deployment(PC 특화 · Target Value)에 전달할 설계 의도.

    모델 이름 · 이 모델에만 있는 CR · Separation 그룹 토론의 합의점 · 이 세그먼트의 EC별 요구 ·
    Judge 판정 사유. 토론 요약 전문(cr_positions 등)은 넣지 않는다.
    """
    is_pro = model_hoq.label == "professional_model"
    segment = settings.SEGMENT_A if is_pro else settings.SEGMENT_B
    sibling = "casual_model" if is_pro else "professional_model"
    own_key, demand_key = ("group_cr_a", "pro_demand") if is_pro else ("group_cr_b", "cas_demand")
    priority = {c["cr"]: c.get("priority", "?") for c in model_hoq.cr_list}

    lines = [f"Model: {model_hoq.label} — the {segment}-purpose model of the product family "
             f"(sibling model: {sibling})."]
    sep_groups = [r for r in conflict_log.get("resolution_log", []) if r.get("strategy") == "Separation"]
    own_crs = [cr for r in sep_groups for cr in r.get(own_key, [])]
    if own_crs:
        lines.append("\nCRs carried only by this model (the sibling model does not carry them):")
        lines += [f"  - [P{priority.get(cr, '?')}] {cr}" for cr in own_crs]
    for gi, r in enumerate(sep_groups, 1):
        summary = r.get("debate_summary", {})
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {}
        agreed = summary.get("points_of_agreement", []) or []
        demands = [(t.get("ec", ""), t.get(demand_key, "")) for t in summary.get("ec_tradeoffs", []) or []
                   if isinstance(t, dict) and t.get(demand_key)]
        lines.append(f"\nSeparation decision (conflict group {gi}):")
        if r.get("reasoning"):
            lines.append(f"  Judge rationale: {r['reasoning']}")
        if agreed:
            lines.append("  Points both segment advocates agreed on:")
            lines += [f"    - {a}" for a in agreed]
        if demands:
            lines.append(f"  {segment} demands on the contested ECs:")
            lines += [f"    - {ec}: {d}" for ec, d in demands]
    return "\n".join(lines)


def save_integrated_hoq(output: OutputSynthesis) -> str:
    """integrated_hoq.json 저장"""
    os.makedirs(settings.OUTPUT_PATH, exist_ok=True)
    path = f"{settings.OUTPUT_PATH}/integrated_hoq.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"  저장: {path}")
    return path


def save_part_deployment(deployment: PartDeployment) -> str:
    """part_deployment.json 저장"""
    os.makedirs(settings.OUTPUT_PATH, exist_ok=True)
    path = f"{settings.OUTPUT_PATH}/part_deployment.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deployment.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"  저장: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: PC Specification (부품 특성 사양서)
# ═════════════════════════════════════════════════════════════════════════════

def build_pc_specifications(
    deployments: list[PartDeploymentResult],
    deployment_raw_results: list[dict],
    output_synthesis: OutputSynthesis,
    conflict_log: dict,
) -> list[PCSpecification]:
    """각 Part Deployment 결과에 대해 PC Specification 생성"""
    resolution_log = conflict_log.get("resolution_log", [])
    conflict_summaries = [
        {
            "group_cr_a": r.get("group_cr_a", []),
            "group_cr_b": r.get("group_cr_b", []),
            "group_cr_common": r.get("group_cr_common", []),
            "strategy": r.get("strategy", ""),
            "reasoning": r.get("reasoning", ""),
            "debate_summary": r.get("debate_summary", ""),
        }
        for r in resolution_log
    ]

    print(f"\n[Phase 3] PC Specification 생성 시작")

    specifications: list[PCSpecification] = []

    for deployment, raw_result in zip(deployments, deployment_raw_results):
        print(f"\n  [{deployment.hoq_label}] PC Specification 생성 중...")

        specs = generate_pc_specification(
            deployment=raw_result,
            strategy=output_synthesis.strategy,
            conflict_summaries=conflict_summaries,
            hoq_label=deployment.hoq_label,
        )

        # LLM 응답 방어 — 키 누락은 스킵+경고, 라벨 불일치는 정규화 시도+경고
        canonical_pcs = {pc_model.pc for pc_model in deployment.pc_list}
        rationale_map: dict[str, str] = {}
        for s in specs:
            pc_key = s.get("pc")
            rationale = s.get("rationale")
            if not pc_key or rationale is None:
                print(f"  ⚠️  [{deployment.hoq_label}] 사양서 항목에 pc/rationale "
                      f"키 누락 — 스킵: {str(s)[:80]}")
                continue
            if pc_key not in canonical_pcs:
                stripped = re.split(r":\s*PI=", pc_key)[0].strip()
                if stripped in canonical_pcs:
                    print(f"  ⚠️  [{deployment.hoq_label}] PC 라벨 정규화: "
                          f"{pc_key[:50]!r} → {stripped[:50]!r}")
                    pc_key = stripped
                else:
                    print(f"  ⚠️  [{deployment.hoq_label}] PC 라벨 불일치 — "
                          f"매칭 실패: {pc_key[:60]!r}")
            rationale_map[pc_key] = rationale

        matched_cnt = sum(1 for pc in canonical_pcs if pc in rationale_map)
        print(f"  [{deployment.hoq_label}] rationale 일치: "
              f"{matched_cnt}/{len(canonical_pcs)}")

        rows: list[PCSpecRow] = []
        for pc_model in deployment.pc_list:
            pc_name = pc_model.pc
            related_ecs = []
            for ec_name, pcs in deployment.ec_pc_relationship_matrix.items():
                score = pcs.get(pc_name, 0)
                if score > 0:
                    related_ecs.append({"ec": ec_name, "score": score})
            related_ecs.sort(key=lambda x: x["score"], reverse=True)

            rows.append(PCSpecRow(
                pc=pc_name,
                part_importance=deployment.part_importance.get(pc_name, 0.0),
                target_value=deployment.target_values.get(pc_name, "N/A"),
                related_ecs=related_ecs,
                rationale=rationale_map.get(pc_name, ""),
            ))

        rows.sort(key=lambda r: r.part_importance, reverse=True)

        missing_rationale = [r.pc for r in rows if not r.rationale]
        if missing_rationale:
            print(f"  ⚠️  [{deployment.hoq_label}] 설정 근거가 비어 있는 PC "
                  f"{len(missing_rationale)}개:")
            for pc in missing_rationale:
                print(f"      - {pc}")

        pc_spec = PCSpecification(
            hoq_label=deployment.hoq_label,
            strategy=output_synthesis.strategy,
            rows=rows,
        )
        specifications.append(pc_spec)

        print(f"  [{deployment.hoq_label}] PC Specification 완료 — {len(rows)} PCs")
        for rank, row in enumerate(rows[:5], 1):
            ecs_str = ", ".join(f"{e['ec']}({e['score']})" for e in row.related_ecs[:3])
            print(f"    {rank}. {row.pc} (PI={row.part_importance:.1f}): {ecs_str}")

    return specifications


def save_pc_specification(specifications: list[PCSpecification]) -> str:
    """pc_specification.xlsx 저장 — HoQ별 시트"""
    os.makedirs(settings.OUTPUT_PATH, exist_ok=True)
    path = f"{settings.OUTPUT_PATH}/pc_specification.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for spec in specifications:
            rows_data = []
            for row in spec.rows:
                ecs_str = ", ".join(
                    f"{e['ec']}({e['score']})" for e in row.related_ecs
                )
                rows_data.append({
                    "PC": row.pc,
                    "Part Importance": row.part_importance,
                    "Target Value": row.target_value,
                    "Related ECs (Relation Score)": ecs_str,
                    "Rationale": row.rationale,
                })

            df = pd.DataFrame(rows_data)
            sheet_name = spec.hoq_label[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  저장: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Part Deployment 후처리
# ═════════════════════════════════════════════════════════════════════════════

def drop_zero_importance_pcs(raw: dict, hoq_label: str) -> dict:
    """Part Importance = 0 인 PC를 산출물에서 제거

    PI = Σ(TI × correlation) 이므로 PI가 0이라는 것은 어떤 EC와도 관계값을
    갖지 않는다는 뜻이다. 임의 판단이 아니라 계산 결과에 근거한 제거이며,
    pc_list / 관계 매트릭스 / roof / target value 를 함께 정리한다.
    """
    part_importance: dict = raw.get("part_importance", {})
    if not part_importance:
        return raw

    keep = {pc for pc, pi in part_importance.items() if (pi or 0) != 0}
    dropped = [pc for pc, pi in part_importance.items() if (pi or 0) == 0]

    # 전량 삭제 방지 가드 — PI가 전부 0이면 상류 결함(EC 키 불일치 등) 신호
    if not keep:
        print(f"  🛑 [{hoq_label}] 모든 PC의 Part Importance가 0입니다 — "
              f"EC-PC 행렬 키 불일치 등 상류 결함이 의심되므로 중단합니다.")
        raise RuntimeError(
            f"[{hoq_label}] PC 전량 삭제 감지 — PI가 전부 0 (상류 결함 의심)"
        )
    if not dropped:
        print(f"  [{hoq_label}] PI=0 PC 없음 — 전체 {len(part_importance)}개 유지")
        return raw

    raw["pc_list"] = [pc for pc in raw.get("pc_list", []) if pc["pc"] in keep]
    raw["part_importance"] = {pc: pi for pc, pi in part_importance.items() if pc in keep}
    raw["target_values"] = {
        pc: tv for pc, tv in raw.get("target_values", {}).items() if pc in keep
    }
    raw["ec_pc_relationship_matrix"] = {
        ec: {pc: v for pc, v in row.items() if pc in keep}
        for ec, row in raw.get("ec_pc_relationship_matrix", {}).items()
    }
    raw["pc_roof_matrix"] = {
        pc_i: {pc_j: v for pc_j, v in row.items() if pc_j in keep}
        for pc_i, row in raw.get("pc_roof_matrix", {}).items()
        if pc_i in keep
    }

    print(f"  [{hoq_label}] PI=0 PC 제거: {len(dropped)}개 "
          f"({len(part_importance)} → {len(keep)})")
    for pc in dropped:
        print(f"    ❌ {pc}")

    return raw


# ═════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인
# ═════════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Stage 3 전체 파이프라인"""
    print("=" * 60)
    print("  Stage 3: Output Synthesis + Part Deployment + PC Specification")
    print("=" * 60)

    # Phase 1: Output Synthesis
    hoq, conflict_log = load_inputs()
    output_synthesis = build_integrated_hoqs(hoq, conflict_log)
    hoq_path = save_integrated_hoq(output_synthesis)

    # Phase 2: Part Deployment
    deployment_raw_results: list[dict] = []
    deployments: list[PartDeploymentResult] = []

    # Separation: 분할 전 전체 HoQ(통합 경로와 같은 TI)로 공통 PC 골격을 먼저 정한다
    is_separation = output_synthesis.strategy == "Separation"
    base_pc_list: list[dict] | None = None
    if is_separation:
        full_imp = _build_full_importance_matrix(hoq, conflict_log)
        full_ti = compute_technical_importance(
            _collect_cr_list(hoq, conflict_log, full_imp), full_imp, hoq.get("ec_list", []),
        )
        base_pc_list = run_pc_skeleton_debate(hoq.get("ec_list", []), full_ti)

    for integrated_hoq in output_synthesis.integrated_hoqs:
        raw = run_part_deployment(
            ec_list=integrated_hoq.ec_list,
            technical_importance=integrated_hoq.technical_importance,
            hoq_label=integrated_hoq.label,
            base_pc_list=base_pc_list,
            model_context=build_model_context(integrated_hoq, conflict_log) if is_separation else None,
        )

        if settings.DROP_ZERO_IMPORTANCE_PC:
            raw = drop_zero_importance_pcs(raw, integrated_hoq.label)

        deployment_raw_results.append(raw)

        deployment = PartDeploymentResult(
            hoq_label=integrated_hoq.label,
            pc_list=[
                PartCharacteristic(pc=pc["pc"], unit=pc.get("unit", ""))
                for pc in raw["pc_list"]
            ],
            ec_pc_relationship_matrix=raw["ec_pc_relationship_matrix"],
            pc_roof_matrix=raw["pc_roof_matrix"],
            part_importance=raw["part_importance"],
            target_values=raw["target_values"],
            target_values_detailed=raw.get("target_values_detailed", {}),
            coverage={**raw.get("coverage", {}),
                      **({"specialization": raw["specialization"]} if raw.get("specialization") else {})},
        )
        deployments.append(deployment)

    full_deployment = PartDeployment(
        output_synthesis=output_synthesis,
        deployments=deployments,
    )
    deploy_path = save_part_deployment(full_deployment)

    # Phase 3: PC Specification
    specifications = build_pc_specifications(
        deployments=deployments,
        deployment_raw_results=deployment_raw_results,
        output_synthesis=output_synthesis,
        conflict_log=conflict_log,
    )
    spec_path = save_pc_specification(specifications)

    # 요약
    print(f"\n{'='*60}")
    print("  Stage 3 완료 요약")
    print("=" * 60)
    print(f"  Strategy: {output_synthesis.strategy}")
    print(f"  Integrated HoQ: {len(output_synthesis.integrated_hoqs)}개 → {hoq_path}")
    print(f"  Part Deployments: {len(deployments)}개 → {deploy_path}")
    for d in deployments:
        print(f"    [{d.hoq_label}] PCs: {len(d.pc_list)}, Target Values: {len(d.target_values)}")
        sorted_pi = sorted(d.part_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 3 PI: ", end="")
        print(", ".join(f"{pc}={pi:.0f}" for pc, pi in sorted_pi[:3]))
    print(f"  PC Specification: {sum(len(s.rows) for s in specifications)}개 PC → {spec_path}")


if __name__ == "__main__":
    run()
