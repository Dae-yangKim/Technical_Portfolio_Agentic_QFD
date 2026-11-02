from pydantic import BaseModel


# ── Stage 3 Phase 1: Output Synthesis ────────────────────────────────────────

class IntegratedHoQ(BaseModel):
    """통합 HoQ — Common + Independent + Adjusted Conflict CRs"""
    label: str  # "integrated" | "professional_model" | "casual_model"
    cr_list: list[dict]  # [{"cr": str, "priority": float, "segment": str, "category": str}]
    ec_list: list[dict]  # [{"ec": str}]
    importance_matrix: dict[str, dict[str, int]]  # {cr: {ec: 0|3|9}}
    direction_matrix: dict[str, dict[str, int]]  # {cr: {ec: -1|0|1}}
    roof_matrix: dict[str, dict[str, float]]  # {ec: {ec: float}}
    technical_importance: dict[str, float]  # {ec: TI score}


class OutputSynthesis(BaseModel):
    """Stage 3 Phase 1 출력"""
    strategy: str  # "Compromise" | "Selection" | "Separation"
    selected_segment: str | None  # Selection 시 선택된 세그먼트
    integrated_hoqs: list[IntegratedHoQ]  # 1개 (Compromise/Selection) or 2개 (Separation)


# ── Stage 3 Phase 2: Part Deployment ─────────────────────────────────────────

class PartCharacteristic(BaseModel):
    """부품 특성(PC) 한 건."""
    pc: str
    unit: str


class PartDeploymentResult(BaseModel):
    """단일 Integrated HoQ에 대한 Part Deployment 결과"""
    hoq_label: str  # 대응하는 IntegratedHoQ.label
    pc_list: list[PartCharacteristic]
    ec_pc_relationship_matrix: dict[str, dict[str, int]]  # {ec: {pc: 0|3|9}}
    pc_roof_matrix: dict[str, dict[str, float]]  # {pc: {pc: float}}
    part_importance: dict[str, float]  # {pc: PI score}
    target_values: dict[str, str]  # {pc: target_value string}
    # {pc: {value, justification, trade_off_notes}} + "_meta"(라운드/합의 기록)
    target_values_detailed: dict[str, dict] = {}
    # 빈 행 검사 기록: pc_debate_uncovered_ecs / empty_ec_rows / ec_pc_fill_meta
    coverage: dict = {}


class PartDeployment(BaseModel):
    """Stage 3 Phase 2 전체 출력"""
    output_synthesis: OutputSynthesis
    deployments: list[PartDeploymentResult]  # 1개 or 2개 (Separation)


# ── Stage 3 Phase 3: PC Specification (부품 특성 사양서) ─────────────────────

class PCSpecRow(BaseModel):
    """PC Specification 테이블의 한 행"""
    pc: str
    part_importance: float
    target_value: str
    related_ecs: list[dict]  # [{"ec": str, "score": int}, ...]
    rationale: str  # 설정 근거


class PCSpecification(BaseModel):
    """하나의 Integrated HoQ에 대한 PC Specification"""
    hoq_label: str
    strategy: str
    rows: list[PCSpecRow]
