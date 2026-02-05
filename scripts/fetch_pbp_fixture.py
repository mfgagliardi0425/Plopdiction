import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv('SPORTRADAR_API_KEY')
if not api_key:
    raise SystemExit('SPORTRADAR_API_KEY not set in environment')

game_id = '1b244e27-f093-40bb-88ec-69b23782cecb'
url = f'https://api.sportradar.com/nba/trial/v8/en/games/{game_id}/pbp.json?api_key={api_key}'
resp = requests.get(url, timeout=60)
print('Status:', resp.status_code)
if resp.status_code != 200:
    print(resp.text[:1000])
    raise SystemExit('Failed to fetch')

out_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sample_pbp.json')
with open(out_path, 'w', encoding='utf-8') as fh:
    json.dump(resp.json(), fh, indent=2)

print('Saved fixture to', out_path)
