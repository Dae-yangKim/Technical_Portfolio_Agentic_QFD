# pipeline
- 서울과학기술대학교 데이터사이언스학과
- 25510099 김대양

**TL;DR: 리뷰 데이터에서 요구사항을 추출해 설계 행렬과 사양서까지 만드는 4단계 LLM 멀티에이전트 파이프라인 코드를 정리하고, 사용한 라이브러리 · 재현을 위한 구성 · 산출물을 이 문서에 구성했다.**

---

### 주요 특징

- 에이전트 간 토론은 LangGraph로 상태 그래프를 구성하였다. 역할이 다른 에이전트가 정해진 최대 라운드 안에서 번갈아 발언하고, 합의하거나 라운드가 끝나면 종료되도록 하였다. 에이전트는 Tavily · SerpAPI · arXiv 검색과 수집 리뷰 검색(ChromaDB)을 도구로 사용한다.
- 의미가 겹치는 항목의 통합은 임베딩과 LLM 두 단계로 구성하였다. OpenAI 임베딩을 ChromaDB에 저장하고 scikit-learn 코사인 유사도로 후보 쌍을 추린 뒤, LLM이 최종 판정하도록 하였다. 입출력 구조는 Pydantic 모델로 정의해 LLM 출력을 검증한다.
- 재현을 위해 의존성은 `uv.lock`과 `requirements.txt`로 버전을 고정하였고, 모델 · 상한값 · 임계값 등 모든 설정은 `settings.py` 한 파일에 모았다. 단계마다 산출물 파일을 남기고 모든 에이전트 토론은 텍스트 로그로 저장한다.

### 사용 라이브러리

| 라이브러리 | 역할 |
|---|---|
| LangGraph | 에이전트 토론 상태 그래프 (노드 · 라운드 · 종료 조건) |
| OpenAI | 에이전트 LLM(gpt-5.1) 호출과 텍스트 임베딩 |
| ChromaDB, LangChain | 리뷰 · 요구사항 임베딩 저장과 유사도 검색 |
| scikit-learn | 임베딩 코사인 유사도 계산 |
| Pydantic | LLM 입출력 데이터 모델 정의와 검증 |
| pandas, openpyxl | 리뷰 입력과 xlsx 산출물 처리 |
| Tavily, SerpAPI, arXiv | 에이전트의 웹 · 논문 검색 도구 |
| rich | 실행 진행 상황과 단계별 소요 시간 출력 |

### 폴더 구조

```
pipeline/
├── main.py                       # Stage 0 → 3 순차 실행
├── settings.py                   # 전역 설정
├── reset_db.py                   # ChromaDB 초기화
├── src/
│   ├── stage0_cr_extraction.py   # 리뷰 태깅 → 요구사항 추출 → 중복 통합 → 우선순위
│   ├── stage1_hoq_construction.py# 기술 특성 도출 → 특성 간 상관 → 요구사항-특성 관계 행렬
│   ├── stage2_conflict_resolution.py  # 충돌 판정 → 그룹화 → 토론 → 판정·조정
│   ├── stage3_spec_deployment.py # 통합 관계 행렬 → 부품 특성 → 특성-부품 행렬 → 목표값 → 사양서
│   ├── prompts.py                # 모든 에이전트 프롬프트
│   ├── agents/debate.py          # LangGraph 토론 그래프
│   └── schemas/                  # Pydantic 데이터 모델 + LangGraph 상태
└── data/                         # 데이터
```

### 설치

```bash
cp ../.env.example ../.env     # OPENAI_API_KEY, TAVILY_API_KEY, SERPAPI_API_KEY 입력
```
저장소 루트의 README.md에서 안내된 내용대로 환경을 만든 뒤 API 키를 설정해서 실행한다.   
입력은 `collection/`으로 수집한 리뷰 파일이며 `data/raw/reddit_reviews.xlsx`에 둔다.

### 사용법

```bash
python main.py               # Stage 0 → 3 전체 실행
python main.py --from 1      # Stage 1부터
python main.py --only 2      # Stage 2만
python reset_db.py           # Stage 0을 다시 돌리기 전 벡터 DB 초기화
```
옵션 정보는 `python main.py --help`로 확인할 수 있다.

### 결과

| Stage | 산출물 | 내용 |
|---|---|---|
| 0 | `data/intermediate/tagged_reviews.xlsx` | 리뷰별 사용 목적 태깅 |
| 0 | `data/intermediate/final_cr_output.xlsx` | 요구사항 목록과 우선순위 |
| 1 | `data/intermediate/hoq.json` | 기술 특성 목록, 기술 특성 간 상관, 요구사항-특성 관계 행렬 |
| 2 | `data/intermediate/conflict_log.json` | 충돌 행렬, 충돌 그룹, 그룹별 전략·근거·토론 요약 |
| 3 | `data/output/integrated_hoq.json` | 조정 반영 Integrated HoQ와 기술적 중요도(TI) |
| 3 | `data/output/part_deployment.json` | 부품 특성 목록, 특성-부품 관계 행렬, 부품 특성 간 상관, 부품 중요도(PI), 목표값 |
| 3 | `data/output/pc_specification.xlsx` | 최종 사양서 |
| 1~3 | `data/intermediate/logs/*.txt` | 에이전트 토론 로그 |