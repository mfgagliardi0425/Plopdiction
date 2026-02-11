import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional


RESULTS_DIR = Path("tracking/results")


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{value:+.{decimals}f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}%"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out_lines = [sep]
    out_lines.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    out_lines.append(sep)
    for row in rows:
        out_lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out_lines.append(sep)
    return "\n".join(out_lines)


def _summarize_payload(payload: dict) -> List[str]:
    summary = payload.get("summary", {})
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pushes = summary.get("pushes", 0)
    return [
        payload.get("date", ""),
        str(summary.get("graded_games", 0)),
        f"{wins}-{losses}-{pushes}",
        _fmt_pct(summary.get("ats_accuracy")),
        _fmt_pct(summary.get("edge_hit_rate")),
        f"{summary.get('model_mae', 0.0):.2f}",
        f"{summary.get('market_mae', 0.0):.2f}",
    ]


def _game_rows(payload: dict) -> List[List[str]]:
    rows = []
    for game in payload.get("games", []):
        metrics = game.get("metrics") or {}
        result = metrics.get("result")
        if not result:
            result = "MISSING"
        rows.append(
            [
                result,
                str(game.get("game", "")),
                _fmt_num(game.get("line_away")),
                _fmt_num(game.get("pred_line_away")),
                _fmt_num(game.get("actual_line_away")),
                _fmt_num(metrics.get("edge")),
            ]
        )
    return rows


def _resolve_payloads(
    file_path: Optional[str],
    single_date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[dict]:
    if file_path:
        return [_load_json(Path(file_path))]

    if single_date:
        target = date.fromisoformat(single_date)
        path = RESULTS_DIR / f"{target.isoformat()}_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        return [_load_json(path)]

    if start_date and end_date:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if end < start:
            raise ValueError("end date must be on or after start date")
        payloads = []
        missing = []
        for day in _iter_dates(start, end):
            path = RESULTS_DIR / f"{day.isoformat()}_results.json"
            if not path.exists():
                missing.append(day.isoformat())
                continue
            payloads.append(_load_json(path))
        if missing:
            print("Missing results for:", ", ".join(missing))
        return payloads

    raise ValueError("Provide --file, --date, or --start/--end.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretty-print ATS results JSON")
    parser.add_argument("--file", help="Path to a results JSON file")
    parser.add_argument("--date", help="Date (YYYY-MM-DD)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--summary-only", action="store_true", help="Print only summary table")
    parser.add_argument("--games-only", action="store_true", help="Print only game-by-game tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only and args.games_only:
        raise ValueError("Choose only one of --summary-only or --games-only")

    payloads = _resolve_payloads(args.file, args.date, args.start, args.end)
    payloads.sort(key=lambda p: p.get("date", ""))

    if not args.games_only:
        summary_rows = [_summarize_payload(p) for p in payloads]
        headers = ["Date", "Graded", "W-L-P", "ATS%", "Edge%", "Model MAE", "Market MAE"]
        print(_render_table(headers, summary_rows))

    if not args.summary_only:
        for payload in payloads:
            print("")
            print(f"Results: {payload.get('date','')}")
            game_headers = ["Res", "Game", "Line(A)", "PredLn(A)", "ActualLn(A)", "Edge"]
            print(_render_table(game_headers, _game_rows(payload)))


if __name__ == "__main__":
    main()
