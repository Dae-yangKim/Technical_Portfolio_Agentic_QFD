# Technical Portfolio
- 서울과학기술대학교 데이터사이언스학과
- 25510099 김대양

**TL;DR: 석사 연구 “An Agentic QFD Framework for Multi-purpose Product Design”의 구현 코드를 테크니컬 포트폴리오 심사 항목(데이터 수집 · 머신러닝 라이브러리 기반 재현 가능한 결과물 · 데이터 시각화)에 맞춰 세 부분으로 나누어 정리하고, 부분별로 README를 구성했다.**

---

### 연구 소개

한 제품을 서로 다른 목적(예: 업무용, 일반용)으로 쓰는 다목적 제품은, 사용 목적에 따라 제품에 요구하는 요구사항이 다르고 때로는 서로 충돌한다.  
예를 들어 노트북에서는 한쪽은 긴 배터리와 가벼운 무게를, 또 다른 쪽은 높은 성능과 낮은 발열 온도를 원하는 식이다.

기존 QFD(Quality Function Deployment)는 단일 고객군을 전제로 고객 요구(CR)를 기술 특성(EC)과 부품 특성(PC)으로 전개하므로, 이런 사용 목적 간 충돌을 다루지 못한다.  
이를 해결하기 위해 사용 목적별로 QFD를 따로 만드는 분리설계가 필요하지만, 전문가의 수작업 판정을 사용 목적의 수만큼 반복해야 하고, 충돌하는 요구를 하나의 설계안으로 통합하는 과정까지 더해지면 시간과 비용이 크게 늘어난다.

그러므로, 이 연구는 agent를 통해 사용 목적별 QFD를 자동 생성하고, 다중 agent 시스템을 통해 생성된 multi QFD를 통합하는 '자동화된 제품 설계 프레임워크'를 제안하는 것을 목적으로 한다.

### 전체 흐름

```
collection/ ──▶ pipeline/ (Stage 0 → 1 → 2 → 3) ──▶ visualization/
리뷰 수집         CR 추출 → HoQ 구축 → 충돌 해결 → 부품 전개       산출물 그림
```

| 단계 | 폴더 | 심사 항목 | 하는 일 | 산출물 |
|---|---|---|---|---|
| 리뷰 수집 | [`collection/`](collection) | 3 (데이터 수집) | Reddit API로 댓글 리뷰 수집 | 리뷰 데이터 |
| Stage 0 · CR 추출 | [`pipeline/`](pipeline) | 7 (머신러닝 라이브러리 기반 재현 가능한 결과물) | 리뷰를 사용 목적별로 태깅하고 CR 추출 및 Priority 산정 | CR 목록 |
| Stage 1 · HoQ 구축 | [`pipeline/`](pipeline) | 7 | 에이전트 토론으로 EC 도출, Roof와 CR-EC 관계 행렬 작성 | hoq.json |
| Stage 2 · 충돌 해결 | [`pipeline/`](pipeline) | 7 | 요구사항 간 충돌 판정, 에이전트 토론과 Judge 판정으로 전략 결정 | conflict_log.json |
| Stage 3 · 부품 전개 | [`pipeline/`](pipeline) | 7 | Integrated HoQ 구성, PC 도출과 EC-PC 행렬, 목표값으로 사양서 작성 | integrated_hoq.json, part_deployment.json, 사양서 |
| 시각화 | [`visualization/`](visualization) | 8 (데이터 시각화) | 산출물 데이터로 HoQ와 각 행렬 그림 생성 | PNG |

### 폴더 구조

```
Technical_portfolio_agentic_qfd/
├── collection/
│   ├── collect_reviews.py        # Reddit 댓글 수집
│   └── README.md
├── pipeline/
│   ├── main.py · settings.py · reset_db.py
│   ├── src/                      # Stage 0~3, 프롬프트, LangGraph 토론 그래프, 스키마
│   ├── data/                     # raw(입력) · intermediate · output (실행 시 생성)
│   └── README.md
├── visualization/
│   ├── render.py                 # 그림 생성
│   ├── sample_input/             # 파이프라인 산출물 예시 json 4개 (clone 직후 실행용)
│   ├── figures/                  # PNG 출력 (실행 시 생성)
│   └── README.md
├── pyproject.toml · uv.lock      # 의존성 (uv)
├── requirements.txt              # 의존성 (pip, uv.lock에서 생성)
├── .python-version               # 3.12
└── .env.example                  # API 키 양식
```

### 설치

**uv**

```bash
git clone https://github.com/Dae-yangKim/Technical_portfolio_agentic_qfd.git
cd Technical_portfolio_agentic_qfd
uv sync
```

**pip**

```bash
git clone https://github.com/Dae-yangKim/Technical_portfolio_agentic_qfd.git
cd Technical_portfolio_agentic_qfd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

필요한 API 키는 `.env.example`을 `.env`로 복사해서 API키를 채운다.

| 키 | 필요한 폴더 | 발급 |
|---|---|---|
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | `collection/` | https://www.reddit.com/prefs/apps ("script" 앱) |
| `OPENAI_API_KEY` | `pipeline/` | OpenAI |
| `TAVILY_API_KEY`, `SERPAPI_API_KEY` | `pipeline/` | Tavily, SerpAPI |
| 없음 | `visualization/` | — |