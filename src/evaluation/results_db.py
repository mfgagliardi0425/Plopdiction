"""Store daily ATS results in a local database (JSON)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path("tracking/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(target_date: date, summary: Dict[str, Any], games: List[Dict[str, Any]]) -> Path:
    payload = {
        "date": target_date.isoformat(),
        "summary": summary,
        "games": games,
    }
    path = RESULTS_DIR / f"{target_date.isoformat()}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
