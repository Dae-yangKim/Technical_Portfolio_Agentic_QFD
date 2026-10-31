# collection — Reddit 리뷰 수집
서울과학기술대학교_데이터사이언스학과_25510099김대양_졸업심사_테크니컬포트폴리오

Reddit 공식 API(PRAW)로 서브레딧의 게시글을 여러 정렬 기준으로 순회하며 댓글을 수집하고, 게시글 ID로 중복을 제거해 xlsx로 저장하는 코드. 수집 결과는 [`../pipeline`](../pipeline) Stage 0의 입력이 된다.

### 주요 특징

- Reddit 공식 API를 PRAW(Python Reddit API Wrapper)로 호출 (읽기 전용)
- 게시글의 댓글을 수집할 때 **복수의 정렬 기준(hot · new · top)을 각각 순회**해 특정 정렬의 상위 주제에 쏠리는 편향을 완화
- **게시글 본문은 제외하고 댓글만 수집**. 본문은 질문·추천 요청이 많고 실제 사용 경험과 의견은 댓글에 담기기 때문
- 같은 게시글이 여러 정렬에 겹쳐 나오면 **게시글 ID 기준으로 한 번만 수집**
- 댓글 트리를 끝까지 펼쳐 **댓글 1건 = 1행**으로 저장. 삭제된 글·일시적 API 오류는 건너뛰고 건수만 보고
- `--since` 옵션으로 **댓글 작성일 기준 기간 선별**. 설계 대상이 향후 출시 제품이므로 현행 세대 제품 사용자의 리뷰만 남기기 위한 것이며, 경계일은 실행 시 지정

### 설치

저장소 루트에서 환경을 만든 뒤(루트 README 참고) Reddit API 키를 설정한다.

```bash
cp ../.env.example ../.env     # REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT 입력
```

키는 https://www.reddit.com/prefs/apps 에서 "script" 타입 앱을 만들면 발급된다.

### 사용법

```bash
python collect_reviews.py                                   # r/laptops, hot·new·top 각 1000개, 전체 저장
python collect_reviews.py --since 2025-01-01                # 2025-01-01 이후 작성된 댓글만 저장
python collect_reviews.py --sorts hot --limit 20 --max-posts 5 --output test.xlsx   # 동작 확인용 소량
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--subreddit` | `laptops` | 수집할 서브레딧 |
| `--sorts` | `hot,new,top` | 순회할 정렬 기준 (쉼표 구분) |
| `--limit` | `1000` | 정렬 기준마다 가져올 게시글 수 (Reddit API 상한 1000) |
| `--time-filter` | `all` | top 정렬의 기간 범위 |
| `--sleep` | `1.2` | 게시글 간 대기 초 (API rate limit 회피) |
| `--replace-more` | `all` | 'load more comments' 펼치기. `all` 또는 정수(`0`이면 펼치지 않음) |
| `--max-posts` | 없음 | 고유 게시글 수 상한 (동작 확인용) |
| `--since` | 없음 | 댓글 작성일이 이 날짜(YYYY-MM-DD) 이후인 리뷰만 저장. 지정하지 않으면 전체 저장 |
| `--output` | `reddit_reviews.xlsx` | 저장 경로 |

### 결과

`reddit_reviews.xlsx` 한 파일에 댓글 1건이 1행으로 저장된다.

| 열 | 설명 |
|---|---|
| `Post_ID` | 댓글이 달린 게시글 ID |
| `Type` | `Comment` (게시글 본문은 수집하지 않음) |
| `Body` | 댓글 본문 |
| `Date` | 댓글 작성 시각 (`YYYY-MM-DD HH:MM:SS`) |
| `URL` | 댓글 permalink |

실행이 끝나면 고유 게시글 수, 댓글 행 수, 건너뛴 게시글 수, 기간 선별 전후 행 수가 화면에 출력된다.

> 수집한 리뷰 데이터는 Reddit 이용약관에 따라 재배포하지 않으므로 이 저장소에 포함하지 않는다.
