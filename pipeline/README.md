# pipeline — Agentic QFD (Stage 0 ~ 3)
서울과학기술대학교_데이터사이언스학과_25510099김대양_졸업심사_테크니컬포트폴리오

사용 목적이 다른 고객 세그먼트의 상충하는 요구사항을 LLM 멀티에이전트 토론으로 통합해 QFD(Quality Function Deployment)를 자동화하는 파이프라인. 리뷰 데이터에서 고객 요구(CR)를 추출하고, 세그먼트별 HoQ(House of Quality)를 구축한 뒤, 충돌하는 요구를 에이전트 토론으로 해결하고, 부품 수준 사양서까지 도출한다.

### 주요 특징

- **4단계 파이프라인**: 리뷰 → CR 추출(Stage 0) → HoQ 구축(Stage 1) → 충돌 해결(Stage 2) → 부품 전개·사양서(Stage 3). 단계마다 독립 실행 가능
- **LangGraph 멀티에이전트 토론**: Engineering ↔ Market 전문가 2-에이전트(HoQ·부품 전개), Segment Advocate ↔ Orchestrator ↔ Judge 4-에이전트(충돌 해결). 합의 또는 최대 라운드에서 종료
- **검색 도구 기반 논증**: 에이전트는 Tavily·Google(SerpAPI)·arXiv·수집 리뷰(ChromaDB) 검색 결과를 근거로 제시
- **임베딩 + LLM 2단 판정**: 세그먼트 간 공통 요구(Common CR)는 OpenAI 임베딩 코사인 유사도로 후보를 추리고 LLM이 최종 판정
- **세 가지 충돌 해결 전략**: Compromise(절충) · Selection(한쪽 우선) · Separation(모델 분리)을 Judge가 결정하고 Importance 행렬을 조정. Separation 시 공통 부품 골격 위에 모델별 사양 도출
- **코드가 형식을, 에이전트가 판단을**: 척도(0/3/9, ±0.5/±0.9)·행렬 구조·라운드 수는 코드가 강제하고, 판정 내용은 에이전트가 결정. 모든 토론은 로그로 저장

### 폴더 구조

```
pipeline/
├── main.py                       # 진입점: Stage 0 → 3 순차 실행 (--from / --to / --only)
├── settings.py                   # 모델·세그먼트·경로·상한값 등 전역 설정
├── reset_db.py                   # ChromaDB 초기화 (Stage 0 재실행 전)
├── src/
│   ├── stage0_cr_extraction.py   # 리뷰 태깅 → 세그먼트별 CR 추출 → Common CR 통합 → Priority
│   ├── stage1_hoq_construction.py# EC 도출 → Roof → CR-EC 관계 행렬 (세그먼트별 HoQ)
│   ├── stage2_conflict_resolution.py  # N×N 충돌 판정 → 충돌 그룹 → 토론 → 전략·조정
│   ├── stage3_spec_deployment.py # Integrated HoQ → PC 도출 → EC-PC 행렬 → 목표값 → 사양서
│   ├── prompts.py                # 모든 에이전트 프롬프트
│   ├── agents/debate.py          # LangGraph 토론 그래프 (EC·Roof·관계행렬·Advocate·Judge·PC·목표값)
│   └── schemas/                  # Pydantic 데이터 모델 + LangGraph 상태
└── data/                         # raw(입력) · intermediate(중간 산출물) · output(최종 산출물)
```

### 설치

저장소 루트에서 환경을 만들고(루트 README 참고) `.env`에 아래 키를 채운다.

| 키 | 용도 |
|---|---|
| `OPENAI_API_KEY` | LLM 호출(gpt-5.1)과 임베딩 |
| `TAVILY_API_KEY`, `SERPAPI_API_KEY` | 에이전트 검색 도구 |

입력 데이터는 [`../collection`](../collection)으로 수집한 리뷰 파일을 `data/raw/reddit_reviews.xlsx`에 둔다.

### 사용법

```bash
cd pipeline
uv run python main.py               # Stage 0 → 3 전체 실행
uv run python main.py --from 1      # Stage 1부터
uv run python main.py --only 2      # Stage 2만
uv run python reset_db.py           # Stage 0을 다시 돌리기 전 벡터 DB 초기화
```

| 옵션 | 설명 |
|---|---|
| `--from N` | 시작 Stage (0~3) |
| `--to N` | 종료 Stage (0~3) |
| `--only N` | 해당 Stage만 실행 |

각 Stage는 이전 Stage의 산출물 파일을 입력으로 읽으므로 순서대로 실행한다. `settings.py`의 주요 설정:

| 설정 | 기본값 | 의미 |
|---|---|---|
| `MODEL` | `gpt-5.1` | 에이전트 LLM |
| `SEGMENT_A`, `SEGMENT_B` | `Professional`, `Casual` | 사용 목적 세그먼트 |
| `MAX_CRS_PER_SEGMENT` | 13 | 세그먼트당 CR 추출 상한 |
| `MAX_ECS`, `MAX_PCS` | 18, 25 | EC·PC 개수 상한 |
| `MAX_DEBATE_ROUNDS`, `MAX_JUDGE_ROUNDS` | 3, 3 | 토론·판정 최대 라운드 |
| `REUSE_TAGGED_REVIEWS` | `True` | 태깅 결과가 있으면 재사용 |

> 전체 실행은 LLM 호출이 많아 수 시간과 API 비용이 든다. 리뷰 수만 건 기준 Stage 0 태깅이 가장 오래 걸린다.

### 결과

| Stage | 산출물 | 내용 |
|---|---|---|
| 0 | `data/intermediate/tagged_reviews.xlsx` | 리뷰별 유효성·사용 목적 태깅 |
| 0 | `data/intermediate/final_cr_output.xlsx` | Common / 세그먼트별 CR 목록과 Priority, 근거 리뷰 |
| 1 | `data/intermediate/hoq.json` | EC 목록, Roof, 세그먼트별 관계·Importance·Direction 행렬 |
| 2 | `data/intermediate/conflict_log.json` | N×N 충돌 행렬, 충돌 그룹, 그룹별 전략·근거·토론 요약, 조정된 Importance |
| 3 | `data/output/integrated_hoq.json` | 조정 반영 Integrated HoQ와 기술적 중요도(TI). Separation 시 모델별 |
| 3 | `data/output/part_deployment.json` | PC 목록, EC-PC 행렬, PC Roof, 부품 중요도(PI), 목표값 |
| 3 | `data/output/pc_specification.xlsx` | 최종 사양서 — PC · PI · 목표값 · 관련 EC · 설정 근거 (모델별 시트) |
| 1~3 | `data/intermediate/logs/*.txt` | 에이전트 토론 로그 |

Stage 1~3 산출물의 예시는 [`../visualization/sample_input`](../visualization/sample_input)에 있으며, [`../visualization`](../visualization)의 `render.py`가 이 파일들로 HoQ·충돌 행렬 등을 그린다.
