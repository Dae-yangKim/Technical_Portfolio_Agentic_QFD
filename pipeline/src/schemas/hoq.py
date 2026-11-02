from pydantic import BaseModel


class HoQ(BaseModel):
    # Relationship Matrix: 방향(부호) × 중요도 크기
    # 음수 = EC가 CR을 저하, 양수 = EC가 CR을 충족
    """세그먼트별 HoQ — CR-EC 관계 행렬과 파생 행렬(Importance · Direction), Roof, 기술적 중요도."""
    relationship_matrix: dict[str, dict[str, int]]  # {cr: {ec: -9|-3|0|3|9}}

    # Importance Matrix: 크기만 (relationship_matrix에서 부호 제거)
    importance_matrix: dict[str, dict[str, int]]    # {cr: {ec: 0|3|9}}

    # Direction Matrix: 방향만 (relationship_matrix에서 크기 제거)
    direction_matrix: dict[str, dict[str, int]]     # {cr: {ec: -1|0|1}}

    # Roof Matrix: EC 간 기술적 상관관계 (연관 없는 쌍은 키 생략)
    roof_matrix: dict[str, dict[str, float]]        # {ec: {ec: -0.9|-0.5|0.5|0.9}}

    # Technical Importance: EC별 기술 중요도 점수
    technical_importance: dict[str, float]           # {ec: TI score}
