# collection
- 서울과학기술대학교 데이터사이언스학과
- 25510099 김대양

**TL;DR: 입력 데이터인 사용자 리뷰를 Reddit 공식 API로 수집하는 코드를 정리하고, 수집 방식 · 설치 · 사용법 · 결과를 이 문서에 구성했다.**

---

### 주요 특징

- 게시글 본문은 질문이나 추천 요청이 대부분이고, 실제 사용 경험이나 주요 의견은 댓글에 많이 있기 때문에 댓글만 수집하도록 코드를 구성하였다. 댓글 트리를 끝까지 펼쳐 댓글 하나를 한 행으로 저장한다.
- Reddit의 정렬 기준 중 hot · new · top 세 가지를 각각 순회해서 수집한다. 한 정렬만 쓰면 상위 몇 개 주제에 쏠리기 때문이다. 수집 후, 겹치는 게시글은 ID로 식별해 중복을 제거한다.
- 수집 결과물은 `xlsx`파일로 저장되며, 이는 `pipeline/` Stage 0의 입력이 된다.

### 설치

```bash
cp ../.env.example ../.env     # REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT 입력
```
저장소 루트의 README.md에서 안내된 내용대로 환경을 만든 뒤 Reddit API 키를 설정해서 실행한다.  
키는 https://www.reddit.com/prefs/apps 에서 발급 가능하다.

### 사용법

```bash
python collect_reviews.py                                   # r/laptops, hot·new·top 각 1000개, 전체 저장
python collect_reviews.py --since 2025-01-01                # 2025-01-01 이후 작성된 댓글만 저장(옵션 사용 예시)
```
옵션 정보는 `python collect_reviews.py --help`로 확인할 수 있다.

### 결과

**reddit_reviews.xlsx** 한 파일에 댓글 1건이 1행으로 저장된다.  
수집된 리뷰 테이블 데이터는 아래와 같은 요소로 구성되어있다.

| 열 | 설명 |
|---|---|
| `Post_ID` | 댓글이 달린 게시글 ID |
| `Type` | `Comment` (게시글 본문은 수집하지 않음) |
| `Body` | 댓글 본문 |
| `Date` | 댓글 작성 시각 (`YYYY-MM-DD HH:MM:SS`) |
| `URL` | 댓글 permalink |