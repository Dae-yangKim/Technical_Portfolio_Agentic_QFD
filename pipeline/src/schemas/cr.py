from pydantic import BaseModel, Field
from typing import Literal


class CR(BaseModel):
    cr: str                                      # 고객 요구사항 문장
    priority: float = Field(ge=1.0, le=5.0)      # 언급 빈도 기반
    segment: Literal["Professional", "Casual"]
