from pydantic import BaseModel


class EC(BaseModel):
    """기술 특성(EC) 한 건."""
    ec: str  # 기술 특성명 (예: "Battery Capacity (Wh)")
