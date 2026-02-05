import os
import requests
import json
import pytest
from dotenv import load_dotenv

load_dotenv()


def _load_data():
    fixture = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_pbp.json')
    if os.getenv('RUN_LIVE_TESTS') == '1':
        api_key = os.getenv('SPORTRADAR_API_KEY')
        if not api_key:
            pytest.skip('RUN_LIVE_TESTS set but SPORTRADAR_API_KEY missing')
        game_id = '1b244e27-f093-40bb-88ec-69b23782cecb'
        url = f'https://api.sportradar.com/nba/trial/v8/en/games/{game_id}/pbp.json?api_key={api_key}'
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            pytest.skip(f'Live API returned {resp.status_code}')
        return resp.json()
    if os.path.exists(fixture):
        with open(fixture, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    pytest.skip('No fixture available and RUN_LIVE_TESTS not enabled')


def test_playbyplay_structure_and_stats():
    data = _load_data()
    assert isinstance(data, dict)
    assert 'periods' in data

    q1 = data['periods'][0]
    assert 'events' in q1
    events = q1['events']
    assert isinstance(events, list)

    # Inspect first few events and ensure statistics are list/dict when present
    for event in events[:5]:
        assert 'event_type' in event
        if 'statistics' in event:
            stats = event['statistics']
            assert isinstance(stats, (list, dict))
