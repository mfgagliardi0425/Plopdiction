"""Send injury report updates to Discord."""
import argparse
from datetime import date
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetching.injury_dataset import update_injury_dataset
from models.injury_adjustment import POINTS_PER_PPG, RANK_MULTIPLIER, ppg_multiplier
from evaluation.discord_notifier import send_discord_message


def parse_args():
    parser = argparse.ArgumentParser(description="Send injury report updates to Discord")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--season", default="2025-26", help="Season for player stats")
    parser.add_argument("--force", action="store_true", help="Bypass ESPN injuries cache")
    parser.add_argument("--full", action="store_true", help="Send full injury list instead of changes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    dataset, changes = update_injury_dataset(target_date, season=args.season, force_refresh=args.force)

    if args.full:
        teams = dataset.get("teams", {}) if isinstance(dataset, dict) else {}
        if not teams:
            print("No injuries found for today.")
            return
        lines = [f"Full injury report for {target_date.isoformat()}:"]
        for team, entries in sorted(teams.items()):
            lines.append("")
            lines.append(f"{team}")
            if not entries:
                lines.append("  (none)")
                continue
            for e in entries:
                player = e.get("player") or ""
                status = (e.get("status") or "").upper() or "UNKNOWN"
                lines.append(f"  - {player} [{status}]")
        # Impact summary: top OUT players by PPG
        impact_rows = []
        for team, entries in teams.items():
            for e in entries:
                status = (e.get("status") or "").lower()
                if status != "out":
                    continue
                ppg = e.get("ppg")
                if ppg is None:
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
            lines.append("Top OUT impact (est. spread pts):")
            for row in impact_rows[:10]:
                lines.append(
                    f"- {row['team']}: {row['player']} ({row['ppg']:.1f} PPG) -> -{row['penalty']:.1f}"
                )

        message = "\n".join(lines)
    else:
        changed = changes.get("players", [])
        if not changed:
            print("No injury changes detected.")
            return
        lines = [f"Injury updates for {target_date.isoformat()}:"]
        lines.extend([f"- {item}" for item in changed])
        message = "\n".join(lines)

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
