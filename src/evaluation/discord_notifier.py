"""Discord webhook notifier."""
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def send_discord_message(content: str, username: Optional[str] = None, webhook_url: Optional[str] = None) -> bool:
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL_GENERAL_INFO") or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("DISCORD_WEBHOOK_URL not set; skipping Discord notification.")
        return False

    payload = {"content": content}
    if username:
        payload["username"] = username

    response = requests.post(url, json=payload, timeout=15)
    if response.status_code >= 400:
        print(f"Discord webhook error: {response.status_code} {response.text}")
        return False
    return True
