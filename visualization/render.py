#!/usr/bin/env python
"""QFD 산출물 시각화 — 파이프라인(Stage 1~3)이 만든 json 4개를 읽어 HoQ·행렬 그림(PNG)을 그린다.

입력 폴더에 다음 파일이 있어야 한다 (파이프라인 산출물 그대로):
  hoq.json             Stage 1  : EC 목록, Roof, CR 종류별 관계·Importance·Direction 행렬
  conflict_log.json    Stage 2  : N×N 충돌 행렬, 충돌 그룹, 조정된 Importance
  integrated_hoq.json  Stage 3  : 조정 반영 Integrated HoQ (Separation 시 모델별)
  part_deployment.json Stage 3  : PC 목록, EC-PC 관계 행렬, PC Roof

사용 예:
  python render.py --input sample_input --out figures
  python render.py --input sample_input --out figures --only initial_hoq,conflict

그림 (키 → 파일):
  seg_hoq        seg_hoq_{common,professional,casual}.png   CR 종류별 HoQ (부호 포함 관계 행렬)
  initial_hoq    initial_hoq.png                            Initial HoQ (전체 CR × EC)
  importance     importance_matrix.png                      Importance Matrix (0/3/9)
  direction      direction_matrix.png                       Direction Matrix (↑/↓)
  roof           roof_matrix.png                            EC Roof (EC 간 상관)
  conflict       conflict_matrix.png                        N×N Conflict Matrix
  integrated_hoq integrated_hoq_<model>.png                 Integrated HoQ, 조정된 셀 강조
  ec_pc          ec_pc_matrix_<model>.png                   EC-PC 관계 행렬
  pc_roof        pc_roof_matrix_<model>.png                 PC Roof (PC 간 상관)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ── 색상 (기존 캡처 규칙) ────────────────────────────────────────────────────────
C = {
    "p9": ("#1d4ed8", "white"), "p3": ("#93c5fd", "black"),
    "n3": ("#fca5a5", "black"), "n9": ("#b91c1c", "white"),
    "r9": ("#15803d", "white"), "r5": ("#86efac", "black"),
    "m5": ("#fdba74", "black"), "m9": ("#c2410c", "white"),
    "up": ("#bbf7d0", "black"), "down": ("#fecaca", "black"),
    "conflict": ("#fde68a", "black"), "skip": ("#f1f5f9", "#94a3b8"),
    "blank": ("white", "black"), "head": ("#f1f5f9", "black"),
}
GRID = "#cbd5e1"
CELL = 0.42          # 행렬 셀 한 변 (inch)
FONT = 7.5

SEG_LABEL = {"common": "Common", "specific_professional": "Pro", "specific_casual": "Cas",
             "Common": "Common", "Professional": "Pro", "Casual": "Cas"}
SEGMENTS = ["common", "specific_professional", "specific_casual"]


# ── 유틸 ──────────────────────────────────────────────────────────────────────────

def shorten(text: str, n: int) -> str:
    """n자에서 말줄임. 끝의 괄호 단위 '(…)'는 가능하면 보존한다."""
    text = text.strip()
    if len(text) <= n:
        return text
    unit = ""
    if text.endswith(")") and "(" in text:
        unit = text[text.rfind("("):]
        if len(unit) > 18:
            unit = ""
    body = text[: max(1, n - len(unit) - 1)].rstrip()
    return f"{body}… {unit}".rstrip() if unit else f"{body}…"


def fmt_num(v) -> str:
    """5.0 → "5", 4.5 → "4.5" """
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def rel_style(v: int) -> tuple[str, str, str]:
    """관계/Importance 값 → (표시 문자, 배경, 글자색)"""
    if v == 0:
        return "", *C["blank"]
    key = {9: "p9", 3: "p3", -3: "n3", -9: "n9"}[v]
    txt = f"+{v}" if v > 0 else f"−{abs(v)}"
    return txt, *C[key]


def imp_style(v: int) -> tuple[str, str, str]:
    """Importance 값(0/3/9)에 따른 셀 배경색 · 글자색 · 표시 문자열."""
    if v == 0:
        return "", *C["blank"]
    return str(v), *C["p9" if v == 9 else "p3"]


def roof_style(v) -> tuple[str, str, str]:
    """Roof 상관값(±0.5/±0.9)에 따른 셀 배경색 · 글자색 · 표시 문자열."""
    if v is None or v == 0:
        return "", *C["blank"]
    key = {0.9: "r9", 0.5: "r5", -0.5: "m5", -0.9: "m9"}[round(float(v), 1)]
    txt = f"+{v}" if v > 0 else f"−{abs(v)}"
    return txt, *C[key]


# ── 공통 그리기 ───────────────────────────────────────────────────────────────────

def draw_table(
    title: str,
    text_cols: list[tuple[str, list[str], float, str]],   # (헤더, 값들, 너비 inch, 정렬 'left'|'center')
    col_labels: list[str],
    cells: list[list[tuple[str, str, str]]],              # [row][col] = (text, facecolor, textcolor)
    out_path: str,
    dpi: int,
    col_widths: list[float] | None = None,
    thick: set[tuple[int, int]] | None = None,            # 굵은 테두리 셀 (row, col)
    label_font: float = FONT,
    footnote: str = "",
) -> None:
    """텍스트 열 + 행렬 셀을 표 형태로 그려 PNG로 저장한다. 셀 크기는 고정, 그림 크기는 행·열 수와 라벨 길이로 계산."""
    n_rows, n_cols = len(cells), len(col_labels)
    widths = col_widths or [CELL] * n_cols
    text_w = sum(w for _, _, w, _ in text_cols)
    grid_w = sum(widths)
    head_h = min(3.6, 0.062 * max((len(s) for s in col_labels), default=4) * label_font / FONT + 0.25)
    title_h, foot_h, pad = 0.45, (0.3 if footnote else 0.0), 0.15
    W = pad * 2 + text_w + grid_w
    H = pad * 2 + title_h + head_h + CELL * n_rows + foot_h

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    x0 = pad
    y_top = H - pad - title_h            # 헤더 영역 위쪽
    y_grid = y_top - head_h              # 행렬 첫 행 위쪽

    ax.text(pad, H - pad - 0.1, title, fontsize=11, fontweight="bold", va="top", ha="left")

    # 텍스트 열 헤더 + 값
    x = x0
    for header, values, w, align in text_cols:
        ax.add_patch(Rectangle((x, y_grid), w, head_h, facecolor=C["head"][0], edgecolor=GRID, lw=0.6))
        ax.text(x + w / 2, y_grid + 0.12, header, fontsize=FONT, ha="center", va="bottom", fontweight="bold")
        for r, val in enumerate(values):
            y = y_grid - CELL * (r + 1)
            ax.add_patch(Rectangle((x, y), w, CELL, facecolor="white", edgecolor=GRID, lw=0.6))
            tx = x + 0.06 if align == "left" else x + w / 2
            ax.text(tx, y + CELL / 2, val, fontsize=FONT, ha=align, va="center")
        x += w

    # 행렬 열 헤더 (세로 회전) + 셀
    gx = x
    for c, lab in enumerate(col_labels):
        w = widths[c]
        ax.add_patch(Rectangle((gx, y_grid), w, head_h, facecolor=C["head"][0], edgecolor=GRID, lw=0.6))
        ax.text(gx + w / 2, y_grid + 0.08, lab, fontsize=label_font, rotation=90, ha="center", va="bottom")
        for r in range(n_rows):
            txt, face, color = cells[r][c]
            y = y_grid - CELL * (r + 1)
            ax.add_patch(Rectangle((gx, y), w, CELL, facecolor=face, edgecolor=GRID, lw=0.6))
            if txt:
                ax.text(gx + w / 2, y + CELL / 2, txt, fontsize=FONT, ha="center", va="center", color=color)
            if thick and (r, c) in thick:
                ax.add_patch(Rectangle((gx, y), w, CELL, fill=False, edgecolor="black", lw=2.0))
        gx += w

    if footnote:
        ax.text(pad, pad + 0.05, footnote, fontsize=FONT - 0.5, color="#475569", va="bottom", ha="left")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out_path}  ({n_rows} × {n_cols})")


# ── 데이터 ────────────────────────────────────────────────────────────────────────

class Data:
    """json 4개를 읽고 CR·EC 번호(CR-n, EC-n)를 매긴다. 번호 순서 = 파이프라인 목록 순서."""

    def __init__(self, input_dir: str):
        def load(name):
            path = os.path.join(input_dir, name)
            if not os.path.exists(path):
                raise SystemExit(f"입력 파일이 없습니다: {path}")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        self.hoq = load("hoq.json")
        self.conflict = load("conflict_log.json")
        self.integrated = load("integrated_hoq.json")
        self.part = load("part_deployment.json")

        self.ecs = [e["ec"] for e in self.hoq["ec_list"]]
        self.ec_id = {ec: f"EC-{i}" for i, ec in enumerate(self.ecs, 1)}
        self.crs: list[dict] = []          # 전체 CR (Common → Pro → Cas), seg 포함
        for seg in SEGMENTS:
            for c in self.hoq[seg]["cr_list"]:
                self.crs.append({**c, "seg": SEG_LABEL[seg]})
        self.cr_id = {c["cr"]: f"CR-{i}" for i, c in enumerate(self.crs, 1)}
        self.cr_info = {c["cr"]: c for c in self.crs}
        self.initial_importance: dict = {}
        for seg in SEGMENTS:
            self.initial_importance.update(self.hoq[seg]["importance_matrix"])

    # 텍스트 열: CR, Seg, VoC, P
    def cr_text_cols(self, crs: list[dict], voc_len: int = 92, with_p: bool = True):
        cols = [
            ("CR", [self.cr_id[c["cr"]] for c in crs], 0.65, "center"),
            ("Seg", [c["seg"] for c in crs], 0.75, "center"),
            ("Customer Requirement (VoC)", [shorten(c.get("voc") or c["cr"], voc_len) for c in crs], 6.3, "left"),
        ]
        if with_p:
            cols.append(("P", [fmt_num(c["priority"]) for c in crs], 0.4, "center"))
        return cols

    def ec_labels(self, n: int = 34) -> list[str]:
        return [f"{self.ec_id[ec]} · {shorten(ec, n)}" for ec in self.ecs]


# ── 그림 ──────────────────────────────────────────────────────────────────────────

def fig_seg_hoq(d: Data, out: str, dpi: int) -> None:
    """CR 종류별(공통 · 세그먼트별) HoQ 관계 행렬을 종류마다 한 장씩 그린다."""
    for seg in SEGMENTS:
        block = d.hoq[seg]
        crs = [{**c, "seg": SEG_LABEL[seg]} for c in block["cr_list"]]
        rm = block["relationship_matrix"]
        cells = [[rel_style(rm[c["cr"]].get(ec, 0)) for ec in d.ecs] for c in crs]
        name = SEG_LABEL[seg].lower().replace("pro", "professional").replace("cas", "casual")
        draw_table(f"Segment HoQ — {SEG_LABEL[seg]} ({len(crs)} CR × {len(d.ecs)} EC, relationship −9…+9)",
                   d.cr_text_cols(crs), d.ec_labels(), cells, os.path.join(out, f"seg_hoq_{name}.png"), dpi)


def fig_initial_hoq(d: Data, out: str, dpi: int) -> None:
    """세 CR 블록을 행 방향으로 쌓은 Initial HoQ(조정 전 관계 행렬)를 그린다."""
    rm = {}
    for seg in SEGMENTS:
        rm.update(d.hoq[seg]["relationship_matrix"])
    cells = [[rel_style(rm[c["cr"]].get(ec, 0)) for ec in d.ecs] for c in d.crs]
    draw_table(f"Initial HoQ — {len(d.crs)} CR × {len(d.ecs)} EC (segment HoQs stacked; relationship −9…+9)",
               d.cr_text_cols(d.crs), d.ec_labels(), cells, os.path.join(out, "initial_hoq.png"), dpi)


def fig_importance(d: Data, out: str, dpi: int) -> None:
    """Importance Matrix(0/3/9)를 그린다."""
    cells = [[imp_style(d.initial_importance[c["cr"]].get(ec, 0)) for ec in d.ecs] for c in d.crs]
    draw_table(f"Importance Matrix — {len(d.crs)} CR × {len(d.ecs)} EC (0 / 3 / 9)",
               d.cr_text_cols(d.crs), d.ec_labels(), cells, os.path.join(out, "importance_matrix.png"), dpi)


def fig_direction(d: Data, out: str, dpi: int) -> None:
    """Direction Matrix(↑/↓)를 그린다."""
    dm = {}
    for seg in SEGMENTS:
        dm.update(d.hoq[seg]["direction_matrix"])
    def style(v):
        if v > 0: return "↑", *C["up"]
        if v < 0: return "↓", *C["down"]
        return "", *C["blank"]
    cells = [[style(dm[c["cr"]].get(ec, 0)) for ec in d.ecs] for c in d.crs]
    draw_table(f"Direction Matrix — {len(d.crs)} CR × {len(d.ecs)} EC (↑ EC increase helps the CR, ↓ hurts)",
               d.cr_text_cols(d.crs), d.ec_labels(), cells, os.path.join(out, "direction_matrix.png"), dpi)


def _roof_cells(items: list[str], roof: dict, id_of: dict) -> list[list[tuple[str, str, str]]]:
    """항목 쌍의 상관값을 상삼각 셀 목록으로 변환한다 (대각선 아래는 빈 셀)."""
    cells = []
    for a in items:
        row = []
        for b in items:
            if a == b:
                row.append(("—", *C["skip"]))
            else:
                v = roof.get(a, {}).get(b, roof.get(b, {}).get(a))
                row.append(roof_style(v))
        cells.append(row)
    return cells


def fig_roof(d: Data, out: str, dpi: int) -> None:
    """EC 간 상관(Roof) 행렬을 그린다."""
    cells = _roof_cells(d.ecs, d.hoq["roof_matrix"], d.ec_id)
    text_cols = [("EC", [d.ec_id[e] for e in d.ecs], 0.65, "center"),
                 ("Engineering Characteristic", [shorten(e, 70) for e in d.ecs], 5.0, "left")]
    n_pairs = sum(1 for r in cells for t in r if t[0] not in ("", "—")) // 2
    draw_table(f"Roof Matrix — EC × EC correlation (±0.5 weak, ±0.9 strong; {n_pairs} correlated pairs)",
               text_cols, [d.ec_id[e] for e in d.ecs], cells, os.path.join(out, "roof_matrix.png"), dpi)


def fig_conflict(d: Data, out: str, dpi: int) -> None:
    """CR × CR 충돌 행렬을 그린다. 충돌 셀에는 근거 EC, 같은 종류 쌍은 × 표시."""
    cm = d.conflict["conflict_matrix"]
    nxn = cm["nxn_matrix"]
    crs = d.crs
    cells, widths = [], []
    texts = [[None] * len(crs) for _ in crs]
    for i, a in enumerate(crs):
        for j, b in enumerate(crs):
            v = nxn.get(a["cr"], {}).get(b["cr"])
            if i == j:
                texts[i][j] = ("—", *C["skip"])
            elif isinstance(v, dict):
                ecs = [d.ec_id[e] for e in v.get("direct", [])] + [d.ec_id[e] for e in v.get("roof", [])]
                texts[i][j] = ("{" + ", ".join(ecs) + "}", *C["conflict"])
            elif isinstance(v, str) or a["seg"] == b["seg"]:
                texts[i][j] = ("×", *C["skip"])
            else:
                texts[i][j] = ("", *C["blank"])
    for j in range(len(crs)):
        longest = max(len(texts[i][j][0]) for i in range(len(crs)))
        widths.append(max(CELL, 0.058 * longest + 0.2))
    n_pairs = sum(1 for i in range(len(crs)) for j in range(len(crs)) if texts[i][j][1] == C["conflict"][0]) // 2
    draw_table(f"N×N Conflict Matrix — {len(crs)} CR ({cm.get('total_conflict_cells', '?')} conflict cells = "
               f"{cm.get('direct_count', '?')} direct, {cm.get('roof_count', '?')} indirect; {n_pairs} CR pairs)",
               d.cr_text_cols(crs, voc_len=72, with_p=False), [d.cr_id[c["cr"]] for c in crs], texts,
               os.path.join(out, "conflict_matrix.png"), dpi, col_widths=widths,
               footnote="— self · × same CR type (not judged) · {EC-k} conflict evidence (EC where the two CRs pull in opposite directions)")


def fig_integrated_hoq(d: Data, out: str, dpi: int) -> None:
    """조정 반영 Integrated HoQ를 모델별로 그린다. Judge가 바꾼 셀은 굵은 테두리와 "전→후"로 강조."""
    for model in d.integrated["integrated_hoqs"]:
        crs = []
        for c in model["cr_list"]:
            info = d.cr_info.get(c["cr"], {})
            crs.append({**c, "seg": SEG_LABEL.get(c.get("segment", ""), info.get("seg", "")),
                        "voc": c.get("voc") or info.get("voc"), "priority": c.get("priority", info.get("priority"))})
        im = model["importance_matrix"]
        cells, thick = [], set()
        n_adj = 0
        for r, c in enumerate(crs):
            row = []
            for k, ec in enumerate(d.ecs):
                new = im.get(c["cr"], {}).get(ec, 0)
                old = d.initial_importance.get(c["cr"], {}).get(ec, 0)
                txt, face, color = imp_style(new)
                if new != old:
                    txt = f"{old}→{new}"
                    thick.add((r, k)); n_adj += 1
                    if new == 0:
                        face, color = "#e2e8f0", "black"
                row.append((txt, face, color))
            cells.append(row)
        label = model.get("label", "integrated")
        draw_table(f"Integrated HoQ — {label} ({len(crs)} CR × {len(d.ecs)} EC; {n_adj} cells adjusted in Stage 2, outlined)",
                   d.cr_text_cols(crs), d.ec_labels(), cells, os.path.join(out, f"integrated_hoq_{label}.png"), dpi, thick=thick,
                   footnote="Outlined cells: importance changed by the Judge (before→after). Rows are the model's CR set after strategy application.")


def fig_ec_pc(d: Data, out: str, dpi: int) -> None:
    """EC × PC 관계 행렬을 모델별로 그린다."""
    for dep in d.part["deployments"]:
        pcs = [p["pc"] for p in dep["pc_list"]]
        pc_id = {pc: f"PC-{i}" for i, pc in enumerate(pcs, 1)}
        mat = dep["ec_pc_relationship_matrix"]
        ecs = [ec for ec in d.ecs if ec in mat] or list(mat.keys())
        cells = [[imp_style(mat.get(ec, {}).get(pc, 0)) for pc in pcs] for ec in ecs]
        text_cols = [("EC", [d.ec_id.get(e, e) for e in ecs], 0.65, "center"),
                     ("Engineering Characteristic", [shorten(e, 70) for e in ecs], 5.0, "left")]
        label = dep["hoq_label"]
        draw_table(f"EC–PC Relationship Matrix — {label} ({len(ecs)} EC × {len(pcs)} PC; 0 / 3 / 9)",
                   text_cols, [f"{pc_id[pc]} · {shorten(pc, 34)}" for pc in pcs], cells,
                   os.path.join(out, f"ec_pc_matrix_{label}.png"), dpi)


def fig_pc_roof(d: Data, out: str, dpi: int) -> None:
    """PC 간 상관(PC Roof) 행렬을 모델별로 그린다."""
    for dep in d.part["deployments"]:
        pcs = [p["pc"] for p in dep["pc_list"]]
        pc_id = {pc: f"PC-{i}" for i, pc in enumerate(pcs, 1)}
        cells = _roof_cells(pcs, dep["pc_roof_matrix"], pc_id)
        text_cols = [("PC", [pc_id[p] for p in pcs], 0.65, "center"),
                     ("Part Characteristic", [shorten(p, 70) for p in pcs], 5.0, "left")]
        n_pairs = sum(1 for r in cells for t in r if t[0] not in ("", "—")) // 2
        label = dep["hoq_label"]
        draw_table(f"PC Roof Matrix — {label} (PC × PC correlation; {n_pairs} correlated pairs)",
                   text_cols, [pc_id[p] for p in pcs], cells, os.path.join(out, f"pc_roof_matrix_{label}.png"), dpi)


FIGURES = {
    "seg_hoq": fig_seg_hoq, "initial_hoq": fig_initial_hoq, "importance": fig_importance,
    "direction": fig_direction, "roof": fig_roof, "conflict": fig_conflict,
    "integrated_hoq": fig_integrated_hoq, "ec_pc": fig_ec_pc, "pc_roof": fig_pc_roof,
}


def main() -> None:
    """CLI 옵션을 읽어 입력 json 4개를 로드하고 지정한 그림을 생성한다."""
    p = argparse.ArgumentParser(description="QFD 산출물 시각화 (hoq / conflict_log / integrated_hoq / part_deployment → PNG)")
    p.add_argument("--input", default="sample_input", help="json 4개가 있는 폴더 (기본 sample_input)")
    p.add_argument("--out", default="figures", help="PNG 저장 폴더 (기본 figures)")
    p.add_argument("--only", default=None, help=f"그릴 그림 키, 쉼표 구분 (기본 전부). 가능: {', '.join(FIGURES)}")
    p.add_argument("--dpi", type=int, default=200, help="PNG 해상도 (기본 200)")
    args = p.parse_args()

    keys = [k.strip() for k in args.only.split(",")] if args.only else list(FIGURES)
    bad = [k for k in keys if k not in FIGURES]
    if bad:
        raise SystemExit(f"알 수 없는 그림 키: {bad} (가능: {', '.join(FIGURES)})")

    d = Data(args.input)
    print(f"[Render] input={args.input} · {len(d.crs)} CR · {len(d.ecs)} EC · figures={keys}")
    for k in keys:
        FIGURES[k](d, args.out, args.dpi)
    print("[Render] done")


if __name__ == "__main__":
    main()
