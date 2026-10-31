from pydantic import BaseModel


class EC(BaseModel):
    ec: str  # 기술 특성명 (예: "Battery Capacity (Wh)")
