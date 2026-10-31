from typing import Annotated, TypedDict
import operator


class QFDState(TypedDict):
    # ── Input ─────────────────────────────────────────
    segment_a: str
    segment_b: str
    cr_list_a: list[dict]
    cr_list_b: list[dict]
    cr_list_common: list[dict]        # Common CR 목록 (priority 조회용 — 대립 강도 프로필)
    separation_reconsidered: bool     # Separation 재고 요청 1회 추적 (Validator)

    # ── Stage 1: CR classification ────────────────────
    common_cr: list[dict]
    specific_cr_a: list[dict]
    specific_cr_b: list[dict]

    # ── Stage 1: EC & HoQ ─────────────────────────────
    ec_list: list[dict]
    importance_matrix: dict
    direction_matrix: dict
    relationship_matrix: dict
    roof_matrix: dict
    phase: str               # "ec" | "roof" — 서브 토론 단계 구분
    coverage_gaps: list[str]  # 발송 9 #22: 빈 행 검사 — EC 토론=미커버 CR, PC 토론=미커버 EC (정본 이름)

    # ── Stage 2: Conflict Resolution ──────────────────
    conflict_matrix: dict
    conflict_log: list[dict]
    current_conflict: dict   # 현재 토론 중인 충돌 그룹 컨텍스트
    resolution_log: list[dict]

    # ── Stage 2: Judge Phase ──────────────────────────
    debate_summary: str      # Advocate 토론의 final summary (Judge 입력)
    judge_strategy: str      # Judge가 결정한 전략
    adjusted_importance: dict  # Judge가 조정한 importance matrix
    selected_segment: str | None   # Selection 전략에서 우선한 세그먼트
    scope_violations: list         # 10/10: 그룹 밖 셀 조정 시도 (Validator가 반려 사유로 사용)
    separated_importance_pro: dict  # Separation 전략의 Professional 모델 importance
    separated_importance_cas: dict  # Separation 전략의 Casual 모델 importance
    judge_failed: bool              # 발송 9: 조정 0건이 끝까지 해소되지 않아 Stage 2를 중단해야 함
    judge_failure_reason: str       # 발송 9: 중단 사유 (로그·예외 메시지용)

    # ── Stage 3: Output Synthesis ─────────────────────
    technical_importance: dict  # {ec: TI score}

    # ── Stage 3: Part Deployment ───────────────────────
    pc_list: list[dict]        # [{"pc": str, "unit": str}]
    base_pc_list: list[dict]   # Separation: 분할 전 HoQ로 정한 공통 PC 골격 (모델별 특화의 출발점)
    model_context: str         # Separation: 설계 중인 모델의 맥락 (전용 CR · Separation 토론 합의점)
    part_characteristics: list[dict]
    target_values: dict

    # ── Debate control ─────────────────────────────────
    iteration: int
    consensus: bool
    messages: Annotated[list, operator.add]
