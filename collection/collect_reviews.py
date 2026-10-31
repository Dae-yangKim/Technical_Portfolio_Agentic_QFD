#!/usr/bin/env python
"""Reddit 리뷰 수집기

서브레딧의 게시글을 복수의 정렬 기준(hot / new / top)으로 각각 순회하며 댓글을 수집하고,
게시글 ID 기준으로 정렬 간 중복을 제거한 뒤 xlsx 로 저장한다.
게시글 본문은 수집하지 않는다. 본문은 질문이나 추천 요청이 많고,
실제 사용 경험과 의견은 댓글에 담기기 때문이다.

사용 예:
  python collect_reviews.py                                  # r/laptops, hot·new·top 각 1000개
  python collect_reviews.py --since 2025-01-01               # 지정일 이후 작성된 댓글만 저장
  python collect_reviews.py --sorts hot --limit 20 --max-posts 5 --output test.xlsx   # 동작 확인용 소량

출력 열: Post_ID, Type, Body, Date, URL   (댓글 1건 = 1행)
인증  : 저장소 루트의 .env 에 REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT (.env.example 참고)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import praw
from dotenv import load_dotenv
from tqdm.auto import tqdm

COLUMNS = ["Post_ID", "Type", "Body", "Date", "URL"]
SORTS = ("hot", "new", "top")


# ── Reddit 클라이언트 ───────────────────────────────────────────────────────────

def make_reddit() -> praw.Reddit:
    """.env 의 자격 증명으로 읽기 전용 Reddit 클라이언트를 만든다."""
    load_dotenv()
    keys = {
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT"),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        sys.exit(f".env 에 다음 값이 없습니다: {', '.join(missing)}  (.env.example 참고)")
    return praw.Reddit(
        client_id=keys["REDDIT_CLIENT_ID"],
        client_secret=keys["REDDIT_CLIENT_SECRET"],
        user_agent=keys["REDDIT_USER_AGENT"],
    )


# ── 수집 ────────────────────────────────────────────────────────────────────────

def iter_submissions(reddit: praw.Reddit, subreddit: str, sort: str, limit: int, time_filter: str):
    """정렬 기준에 맞는 게시글 제너레이터를 돌려준다."""
    sub = reddit.subreddit(subreddit)
    if sort == "hot":
        return sub.hot(limit=limit)
    if sort == "new":
        return sub.new(limit=limit)
    if sort == "top":
        return sub.top(limit=limit, time_filter=time_filter)
    raise ValueError(f"지원하지 않는 정렬: {sort}")


def collect_comments(submission, replace_more: int | None) -> list[dict]:
    """게시글 하나의 댓글 트리를 펼쳐 행 목록으로 만든다 (본문은 제외).

    replace_more=None 이면 'load more comments' 를 전부 펼친다 (느리지만 완전).
    0 이면 펼치지 않고 처음 로드된 댓글만 가져온다 (빠르지만 일부 누락).
    """
    submission.comments.replace_more(limit=replace_more)
    rows = []
    for comment in submission.comments.list():
        rows.append({
            "Post_ID": submission.id,
            "Type": "Comment",
            "Body": comment.body,
            "Date": datetime.fromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "URL": f"https://www.reddit.com{comment.permalink}",
        })
    return rows


def collect(
    reddit: praw.Reddit,
    subreddit: str,
    sorts: list[str],
    limit: int,
    time_filter: str = "all",
    sleep: float = 1.2,
    replace_more: int | None = None,
    max_posts: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """복수의 정렬 기준을 순서대로 순회하며, 게시글 ID 기준으로 중복 없이 댓글을 모은다.

    정렬 기준 하나만 쓰면 그 기준의 상위 게시글 주제에 편중되므로,
    hot / new / top 을 각각 순회해 주제 편향을 완화한다.
    같은 게시글이 여러 정렬에 겹쳐 나오면 처음 한 번만 수집한다.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    skipped = 0

    for sort in sorts:
        gen = iter_submissions(reddit, subreddit, sort, limit, time_filter)
        for submission in tqdm(gen, total=limit, desc=f"{sort:<4}"):
            if submission.id in seen:
                continue
            seen.add(submission.id)
            try:
                rows.extend(collect_comments(submission, replace_more))
            except Exception as e:  # 삭제된 글·일시적 API 오류는 건너뛴다
                skipped += 1
                print(f"  [skip] {submission.id}: {e}")
            time.sleep(sleep)  # 429 Too Many Requests 방지
            if max_posts and len(seen) >= max_posts:
                break
        if max_posts and len(seen) >= max_posts:
            break

    df = pd.DataFrame(rows, columns=COLUMNS)
    summary = {"unique_posts": len(seen), "rows": len(df), "skipped_posts": skipped}
    return df, summary


# ── 기간 기준 선별 ───────────────────────────────────────────────────────────────

def filter_by_date(df: pd.DataFrame, since: str) -> pd.DataFrame:
    """댓글 작성일 기준 기간 선별

    설계 대상이 향후 출시 제품이므로, 현행 세대 제품을 사용하며 작성된 리뷰만을 대상으로 한다.
    댓글에는 사용 제품이 명시되지 않는 경우가 많아 제품 기준 분류가 불가능하므로,
    모든 댓글에 존재하는 작성일을 기준으로 기간을 한정한다.
    경계일은 플랫폼 공급사의 세대 교체 시점을 근거로 정하며, 실행 시 --since 로 지정한다.
    """
    dates = pd.to_datetime(df["Date"])
    return df[dates >= pd.Timestamp(since)].reset_index(drop=True)


# ── 저장 ────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, output: str) -> None:
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_excel(output, index=False)


# ── 진입점 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Reddit 서브레딧 댓글 수집기 (PRAW)")
    p.add_argument("--subreddit", default="laptops", help="수집할 서브레딧 (기본 laptops)")
    p.add_argument("--sorts", default="hot,new,top", help="정렬 기준, 쉼표 구분 (기본 hot,new,top)")
    p.add_argument("--limit", type=int, default=1000, help="정렬 기준마다 가져올 게시글 수 (기본 1000, API 상한)")
    p.add_argument("--time-filter", default="all", choices=["all", "year", "month", "week", "day", "hour"],
                   help="top 정렬의 기간 (기본 all)")
    p.add_argument("--sleep", type=float, default=1.2, help="게시글 간 대기 초 (기본 1.2)")
    p.add_argument("--replace-more", default="all",
                   help="'load more comments' 펼치기: all(전부) 또는 정수 (0이면 펼치지 않음)")
    p.add_argument("--max-posts", type=int, default=None, help="고유 게시글 수 상한 (동작 확인용)")
    p.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                   help="댓글 작성일이 이 날짜 이후인 리뷰만 저장 (지정하지 않으면 전체 저장)")
    p.add_argument("--output", default="reddit_reviews.xlsx", help="저장 경로 (기본 reddit_reviews.xlsx)")
    args = p.parse_args()

    sorts = [s.strip() for s in args.sorts.split(",") if s.strip()]
    bad = [s for s in sorts if s not in SORTS]
    if bad:
        sys.exit(f"지원하지 않는 정렬: {bad} (가능: {', '.join(SORTS)})")
    replace_more = None if args.replace_more == "all" else int(args.replace_more)
    if args.since:
        try:
            pd.Timestamp(args.since)
        except ValueError:
            sys.exit(f"--since 는 YYYY-MM-DD 형식이어야 합니다: {args.since}")

    print(f"[Collect] r/{args.subreddit} · sorts={sorts} · limit={args.limit}/sort · "
          f"replace_more={args.replace_more} · sleep={args.sleep}s")
    reddit = make_reddit()
    df, summary = collect(reddit, args.subreddit, sorts, args.limit, args.time_filter,
                          args.sleep, replace_more, args.max_posts)
    print(f"[Collect] 고유 게시글 {summary['unique_posts']:,}개 · 댓글 {summary['rows']:,}행 · "
          f"건너뜀 {summary['skipped_posts']}개")

    if args.since:
        before = len(df)
        df = filter_by_date(df, args.since)
        print(f"[Filter] 작성일 {args.since} 이후: {before:,} → {len(df):,}행")

    save(df, args.output)
    print(f"[Save] {args.output} ({len(df):,}행)")


if __name__ == "__main__":
    main()
