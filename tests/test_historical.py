import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('SPORTRADAR_API_KEY')
game_id = '1b244e27-f093-40bb-88ec-69b23782cecb'
url = f'https://api.sportradar.com/nba/trial/v8/en/games/{game_id}/summary.json?api_key={api_key}'

response = requests.get(url, timeout=30)
print(f'Status: {response.status_code}')

data = response.json()
print(f'Game: {data.get("away", {}).get("alias")} @ {data.get("home", {}).get("alias")}')
print(f'Final Score: {data.get("away", {}).get("points")} - {data.get("home", {}).get("points")}')
print(f'Status: {data.get("status")}')
print(f'Has detailed stats: {"statistics" in data.get("home", {})}')

# Check what player data is available
if "home" in data and "players" in data["home"]:
    print(f'Number of home players: {len(data["home"]["players"])}')
    if data["home"]["players"]:
        player = data["home"]["players"][0]
        print(f'Example player stats keys: {list(player.get("statistics", {}).keys())}')
