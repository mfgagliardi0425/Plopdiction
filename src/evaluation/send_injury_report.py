"""Send injury report updates to Discord."""
import argparse
from datetime import date, datetime, time
from pathlib import Path
import os
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.espn_api import get_espn_games_for_date
from data_fetching.injury_dataset import update_injury_dataset
from models.injury_adjustment import POINTS_PER_PPG, RANK_MULTIPLIER, ppg_multiplier
from evaluation.discord_notifier import send_discord_message


def parse_args():
    parser = argparse.ArgumentParser(description="Send injury report updates to Discord")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--season", default="2025-26", help="Season for player stats")
    parser.add_argument("--force", action="store_true", help="Bypass ESPN injuries cache")
    parser.add_argument("--full", action="store_true", help="Send full injury list instead of changes")
    parser.add_argument("--changes", action="store_true", help="Send only changes instead of full report")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending")
    parser.add_argument(
        "--min-start-est",
        default=None,
        help="Only include teams with games at/after this EST time (HH:MM)",
    )
    parser.add_argument(
        "--remaining",
        action="store_true",
        help="Only include teams with games not in-progress or completed",
    )
    return parser.parse_args()


def _parse_est_cutoff(value: str | None) -> time | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError:
        raise ValueError("--min-start-est must be HH:MM (24h format)")


def _game_at_or_after_cutoff(game: dict, cutoff: time) -> bool:
    start_time = game.get("start_time_utc")
    if not start_time:
        return True
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError:
        return True
    est_dt = start_dt.astimezone(ZoneInfo("America/New_York"))
    return est_dt.time() >= cutoff


def _is_remaining_game(game: dict) -> bool:
    status = game.get("status")
    if not isinstance(status, dict):
        return True
    if status.get("completed") is True:
        return False
    state = status.get("state")
    if state and state != "pre":
        return False
    return True


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    games = get_espn_games_for_date(target_date)
    cutoff = _parse_est_cutoff(args.min_start_est)
    if cutoff is not None:
        games = [g for g in games if _game_at_or_after_cutoff(g, cutoff)]
    if args.remaining:
        games = [g for g in games if _is_remaining_game(g)]
    teams_playing = {g["home_team"] for g in games} | {g["away_team"] for g in games}

    dataset, changes = update_injury_dataset(target_date, season=args.season, force_refresh=args.force)

    teams = dataset.get("teams", {}) if isinstance(dataset, dict) else {}
    if not teams:
        print("No injuries found for today.")
        return

    send_full = True
    if args.changes:
        send_full = False
    if args.full:
        send_full = True

    if send_full:
        report_teams = {team: entries for team, entries in teams.items() if team in teams_playing}
        report_title = f"Injury report for {target_date.isoformat()}:"
    else:
        changed_teams = set(changes.get("teams", []))
        if not changed_teams:
            print("No injury changes detected.")
            return
        report_teams = {
            team: teams.get(team, [])
            for team in changed_teams
            if team in teams_playing
        }
        report_title = f"Injury updates for {target_date.isoformat()}:"

    if not report_teams:
        print("No injury updates for teams playing today.")
        return

    lines = [report_title, ""]
    for team, entries in sorted(report_teams.items()):
        if not entries:
            lines.append(f"{team}: (none)")
            continue
        players = []
        for e in entries:
            player = e.get("player") or ""
            status = (e.get("status") or "").upper() or "UNKNOWN"
            players.append(f"{player} ({status})")
        lines.append(f"{team}: {', '.join(players)}")

    # Impact summary: top OUT impact for players above the PPG threshold
    impact_rows = []
    for team, entries in report_teams.items():
        for e in entries:
            status = (e.get("status") or "").lower()
            if status != "out":
                continue
            ppg = e.get("ppg")
            if ppg is None:
                continue
            if float(ppg) < 15.0:
                continue
            rank = e.get("ppg_rank")
            mult = RANK_MULTIPLIER.get(rank, 1.0) if rank else ppg_multiplier(float(ppg))
            penalty = float(ppg) * POINTS_PER_PPG * mult
            impact_rows.append({
                "team": team,
                "player": e.get("player") or "",
                "ppg": float(ppg),
                "penalty": penalty,
            })

    if impact_rows:
        impact_rows.sort(key=lambda x: x["ppg"], reverse=True)
        lines.append("")
        lines.append("Top player impact (OUT, >= 15.0 PPG, est. spread pts):")
        for row in impact_rows[:10]:
            lines.append(
                f"- {row['team']}: {row['player']} ({row['ppg']:.1f} PPG) -> -{row['penalty']:.1f}"
            )

    message = "\n".join(lines)

    if args.dry_run:
        print(message)
        return

    webhook = os.getenv("DISCORD_WEBHOOK_URL_INJURY_REPORT")
    if len(message) <= 1900:
        sent = send_discord_message(message, username="Injury Report", webhook_url=webhook)
        if sent:
            print("Sent injury report update.")
        else:
            print("Injury report not sent.")
        return

    # Chunk long messages for Discord 2000-char limit
    chunks = []
    current = ""
    for line in message.split("\n"):
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) > 1900:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)

    for idx, chunk in enumerate(chunks, 1):
        header = f"Injury Report ({idx}/{len(chunks)}):"
        send_discord_message(f"{header}\n{chunk}", username="Injury Report", webhook_url=webhook)
    print("Sent injury report update in chunks.")


if __name__ == "__main__":
    main()
