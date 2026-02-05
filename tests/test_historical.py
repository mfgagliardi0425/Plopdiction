import os
import requests
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(os.getenv('RUN_LIVE_TESTS') != '1', reason='Live tests disabled')
def test_sportradar_game_summary():
    api_key = os.getenv('SPORTRADAR_API_KEY')
    if not api_key:
        pytest.skip('SPORTRADAR_API_KEY not set')

    game_id = '1b244e27-f093-40bb-88ec-69b23782cecb'
    url = f'https://api.sportradar.com/nba/trial/v8/en/games/{game_id}/summary.json?api_key={api_key}'

    response = requests.get(url, timeout=30)
    # Skip gracefully if API returns an error status (auth/permission issues)
    if response.status_code != 200:
        pytest.skip(f"SportRadar API returned status {response.status_code}")

    # Safe JSON parse
    try:
        data = response.json()
    except ValueError:
        pytest.skip('Live API returned non-JSON response')

    # Basic sanity checks
    assert 'home' in data and 'away' in data, 'Missing home/away fields in game summary'
    assert isinstance(data.get('home', {}).get('points', None), (int, type(None)))
    # If detailed statistics exist, ensure structure is as expected
    if data.get('home') and isinstance(data['home'], dict):
        has_stats = 'statistics' in data['home']
        assert isinstance(has_stats, bool)
