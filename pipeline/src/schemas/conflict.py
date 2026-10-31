from pydantic import BaseModel
from typing import Literal


class ConflictCell(BaseModel):
    """N×N Conflict Matrix의 개별 충돌 셀

    - direct: 동일 EC에서 두 세그먼트 CR이 반대 방향
    - roof:   각각 다른 EC를 양의 방향으로 원하지만 해당 EC 쌍이 음의 상관관계
    """
    cr_a: str                                # Segment A의 CR
    cr_b: str                                # Segment B의 CR
    conflict_type: Literal["direct", "roof"]
    ec: str | None = None                    # direct 충돌 시 해당 EC
    ec_a: str | None = None                  # roof 충돌 시 CR_A 측 EC
    ec_b: str | None = None                  # roof 충돌 시 CR_B 측 EC


class ConflictGroup(BaseModel):
    """EC 기준으로 묶인 충돌 그룹 (10/10): 같은 EC(들)를 두고 다투는 충돌 셀과 그 CR들

    하나의 CR이 여러 상대 CR과 충돌하는 경우 관련 pair를 모두 묶어 하나의 토론 단위로 구성
    """
    cr_a_list: list[str]                     # 이 그룹에 속하는 Segment A CR들
    cr_b_list: list[str]                     # 이 그룹에 속하는 Segment B CR들
    cr_common_list: list[str] = []           # 이 그룹에 속하는 Common CR들
    cells: list[ConflictCell]                # 이 그룹에 속하는 모든 충돌 셀
    ec_list: list[str] = []                  # 이 그룹이 다투는 EC 목록 (10/10 EC 기준 그룹화)


class ResolutionLog(BaseModel):
    """충돌 그룹별 해결 결과"""
    group_cr_a: list[str]
    group_cr_b: list[str]
    group_cr_common: list[str] = []
    strategy: Literal["Compromise", "Selection", "Separation"]
    selected_segment: str | None = None
    reasoning: str
    debate_summary: str
    judge_discussion_log: str
    adjusted_importance: dict[str, dict[str, int]]
