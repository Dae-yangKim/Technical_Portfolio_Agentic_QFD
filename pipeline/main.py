"""
Agentic QFD Framework — Full Pipeline Runner

Usage:
    python main.py                  # Stage 0 → 1 → 2 → 3 전체 실행
    python main.py --from 1         # Stage 1부터 실행 (0 생략)
    python main.py --from 2         # Stage 2부터 실행 (0, 1 생략)
    python main.py --only 3         # Stage 3만 실행
"""
import argparse
import time
import sys
import os

import settings

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.live import Live

console = Console()


# ── Stage Metadata ──────────────────────────────────────────────────────────

STAGES = {
    0: {
        "label": "CR Extraction",
        "emoji": "📋",
        "desc": "Reddit 수집 → Purpose Tagging → CR 추출",
        "color": "cyan",
    },
    1: {
        "label": "HoQ Construction",
        "emoji": "🏗️",
        "desc": "EC 토론 → Roof 토론 → Matrix Fill → HoQ 구성",
        "color": "green",
    },
    2: {
        "label": "Conflict Resolution",
        "emoji": "⚔️",
        "desc": "Conflict Matrix → Advocate 토론 → Judge 판결",
        "color": "yellow",
    },
    3: {
        "label": "Output Synthesis + Part Deployment",
        "emoji": "🎯",
        "desc": "Integrated HoQ + TI → PC 토론 → EC-PC Matrix → Target Values",
        "color": "magenta",
    },
}


# ── Utilities ───────────────────────────────────────────────────────────────

def _elapsed_str(secs: float) -> str:
    """경과 시간을 M분 S초 형식으로 반환"""
    m, s = divmod(int(secs), 60)
    if m > 0:
        return f"{m}분 {s}초"
    return f"{s}초"


def _check_file(path: str, label: str) -> None:
    """파일 존재 확인 — 없으면 에러"""
    if not os.path.exists(path):
        console.print(f"\n  [bold red]❌ {label} 파일이 없습니다:[/] {path}")
        console.print(f"     이전 Stage를 먼저 실행해주세요.\n")
        sys.exit(1)


def _file_status(path: str) -> str:
    """파일 존재 여부 아이콘"""
    return "✅" if os.path.exists(path) else "⬜"


# ── Stage Runners ───────────────────────────────────────────────────────────

def run_stage_0() -> None:
    """입력 파일을 확인하고 Stage 0(CR 추출)을 실행한다."""
    _check_file(settings.RAW_DATA_PATH, "Raw data")
    from src.stage0_cr_extraction import run
    run()


def run_stage_1() -> None:
    """Stage 0 산출물을 확인하고 Stage 1(HoQ 구축)을 실행한다."""
    _check_file(
        f"{settings.INTERMEDIATE_PATH}/final_cr_output.xlsx",
        "Stage 0 output (final_cr_output.xlsx)",
    )
    from src.stage1_hoq_construction import run
    run()


def run_stage_2() -> None:
    """Stage 1 산출물을 확인하고 Stage 2(충돌 해결)를 실행한다."""
    _check_file(
        f"{settings.INTERMEDIATE_PATH}/hoq.json",
        "Stage 1 output (hoq.json)",
    )
    from src.stage2_conflict_resolution import run
    run()


def run_stage_3() -> None:
    """Stage 1·2 산출물을 확인하고 Stage 3(부품 전개·사양서)을 실행한다."""
    _check_file(
        f"{settings.INTERMEDIATE_PATH}/hoq.json",
        "Stage 1 output (hoq.json)",
    )
    _check_file(
        f"{settings.INTERMEDIATE_PATH}/conflict_log.json",
        "Stage 2 output (conflict_log.json)",
    )
    from src.stage3_spec_deployment import run
    run()


STAGE_RUNNERS = {
    0: run_stage_0,
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
}


# ── Pipeline Orchestrator ───────────────────────────────────────────────────

def _print_header(stage_from: int, stage_to: int) -> None:
    """파이프라인 시작 헤더 출력"""
    title = Text("🤖 Agentic QFD Framework", style="bold white")

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="bold cyan", width=16)
    info_table.add_column()
    info_table.add_row("🧠 Model",       settings.MODEL)
    info_table.add_row("⚡ Service Tier", settings.SERVICE_TIER)
    info_table.add_row("👥 Segments",     f"{settings.SEGMENT_A} vs {settings.SEGMENT_B}")
    info_table.add_row("📂 Raw Data",     settings.RAW_DATA_PATH)
    info_table.add_row("📁 Intermediate", settings.INTERMEDIATE_PATH)
    info_table.add_row("📁 Output",       settings.OUTPUT_PATH)

    stage_items = []
    for sn in range(stage_from, stage_to + 1):
        meta = STAGES[sn]
        stage_items.append(f"{meta['emoji']} Stage {sn}: {meta['label']}")
    stages_text = "\n".join(stage_items)

    info_table.add_row("🎬 Pipeline", stages_text)

    console.print()
    console.print(Panel(info_table, title=title, border_style="bright_blue", width=74))
    console.print()


def _print_stage_banner(stage_num: int) -> None:
    """Stage 시작 배너"""
    meta = STAGES[stage_num]
    title = f"{meta['emoji']}  Stage {stage_num}: {meta['label']}"
    console.print()
    console.print(Rule(title, style=meta["color"]))
    console.print(f"  [dim]{meta['desc']}[/]")
    console.print()


def _print_summary(
    stage_times: dict[int, float],
    pipeline_elapsed: float,
) -> None:
    """파이프라인 완료 요약"""
    console.print()
    console.print(Rule("🎉 PIPELINE COMPLETE", style="bold green"))
    console.print()

    time_table = Table(
        title="⏱️  Stage별 소요 시간",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    time_table.add_column("#", style="bold", width=3)
    time_table.add_column("Stage", width=44)
    time_table.add_column("소요 시간", justify="right", width=12)
    time_table.add_column("Status", justify="center", width=8)

    for sn in sorted(stage_times):
        meta = STAGES[sn]
        time_str = _elapsed_str(stage_times[sn])
        time_table.add_row(
            str(sn),
            f"{meta['emoji']} {meta['label']}",
            time_str,
            "✅",
        )

    console.print(time_table)
    console.print(f"\n  ⏱️  총 소요 시간: [bold]{_elapsed_str(pipeline_elapsed)}[/]")

    console.print()
    file_table = Table(
        title="📦 Output Files",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    file_table.add_column("Status", width=6, justify="center")
    file_table.add_column("Stage", width=3)
    file_table.add_column("File Path")

    output_files = [
        (0, f"{settings.INTERMEDIATE_PATH}/tagged_reviews.xlsx"),
        (0, f"{settings.INTERMEDIATE_PATH}/final_cr_output.xlsx"),
        (1, f"{settings.INTERMEDIATE_PATH}/hoq.json"),
        (2, f"{settings.INTERMEDIATE_PATH}/conflict_log.json"),
        (3, f"{settings.OUTPUT_PATH}/integrated_hoq.json"),
        (3, f"{settings.OUTPUT_PATH}/part_deployment.json"),
    ]

    for stage_n, path in output_files:
        status = _file_status(path)
        file_table.add_row(status, str(stage_n), path)

    console.print(file_table)

    log_dir = f"{settings.INTERMEDIATE_PATH}/logs"
    if os.path.exists(log_dir):
        log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".txt"))
        if log_files:
            console.print(f"\n  📝 토론 로그: [dim]{log_dir}/[/] ({len(log_files)}개)")
            for lf in log_files:
                size_kb = os.path.getsize(os.path.join(log_dir, lf)) / 1024
                console.print(f"     [dim]└─[/] {lf} [dim]({size_kb:.1f} KB)[/]")

    console.print()


def run_pipeline(stage_from: int = 0, stage_to: int = 3) -> None:
    """지정 범위의 Stage를 순차 실행"""
    pipeline_start = time.time()
    total_stages = stage_to - stage_from + 1

    _print_header(stage_from, stage_to)

    # ── 전체 Progress Bar ──
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )

    pipeline_task = progress.add_task(
        "Pipeline Progress", total=total_stages,
    )

    stage_times: dict[int, float] = {}

    progress.start()

    for stage_num in range(stage_from, stage_to + 1):
        if stage_num not in STAGES:
            console.print(f"\n  [yellow]⚠️  Stage {stage_num} 은 존재하지 않습니다. 건너뜁니다.[/]")
            progress.advance(pipeline_task)
            continue

        meta = STAGES[stage_num]
        progress.update(
            pipeline_task,
            description=f"{meta['emoji']} Stage {stage_num}: {meta['label']}",
        )

        # 배너는 progress bar 밖에서 출력
        progress.stop()
        _print_stage_banner(stage_num)
        progress.start()

        stage_start = time.time()

        try:
            # Stage 실행 중에는 progress bar를 일시 정지 (Stage 내부 print와 겹치지 않도록)
            progress.stop()
            STAGE_RUNNERS[stage_num]()
            progress.start()
        except Exception as e:
            progress.stop()
            console.print(f"\n  [bold red]❌ Stage {stage_num} ({meta['label']}) 실패:[/] {e}")
            console.print(f"     경과 시간: {_elapsed_str(time.time() - pipeline_start)}\n")
            raise

        elapsed = time.time() - stage_start
        stage_times[stage_num] = elapsed

        progress.stop()
        console.print(
            f"\n  [green]✅ Stage {stage_num} 완료[/] — "
            f"[bold]{_elapsed_str(elapsed)}[/]"
        )
        progress.start()

        progress.advance(pipeline_task)

    progress.update(pipeline_task, description="[bold green]🎉 All Stages Complete")
    progress.stop()

    # ── 최종 요약 ──
    pipeline_elapsed = time.time() - pipeline_start
    _print_summary(stage_times, pipeline_elapsed)


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    """CLI 옵션(--from / --to / --only)을 해석해 지정 범위의 Stage를 순차 실행한다."""
    parser = argparse.ArgumentParser(
        description="🤖 Agentic QFD Framework — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Stage 0 → 3 전체 실행
  python main.py --from 1         # Stage 1부터 끝까지
  python main.py --from 2 --to 2  # Stage 2만 실행
  python main.py --only 3         # Stage 3만 실행
        """,
    )
    parser.add_argument(
        "--from", dest="stage_from", type=int, default=0,
        help="시작 Stage 번호 (default: 0)",
    )
    parser.add_argument(
        "--to", dest="stage_to", type=int, default=3,
        help="종료 Stage 번호 (default: 3)",
    )
    parser.add_argument(
        "--only", dest="only", type=int, default=None,
        help="특정 Stage만 실행",
    )

    args = parser.parse_args()

    if args.only is not None:
        stage_from = args.only
        stage_to = args.only
    else:
        stage_from = args.stage_from
        stage_to = args.stage_to

    if stage_from < 0 or stage_to > 3 or stage_from > stage_to:
        console.print(
            f"  [bold red]❌ 잘못된 Stage 범위:[/] "
            f"{stage_from} → {stage_to} (유효 범위: 0~3)"
        )
        sys.exit(1)

    run_pipeline(stage_from, stage_to)


if __name__ == "__main__":
    main()