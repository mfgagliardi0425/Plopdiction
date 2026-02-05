import os
import requests
import json
import pytest
from dotenv import load_dotenv

load_dotenv()

def _load_data():
    # Prefer local fixture for deterministic test runs
    fixture = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_pbp.json')
    if os.getenv('RUN_LIVE_TESTS') == '1':
        api_key = os.getenv('SPORTRADAR_API_KEY')
        if not api_key:
            pytest.skip('RUN_LIVE_TESTS set but SPORTRADAR_API_KEY missing')
        game_id = '1b244e27-f093-40bb-88ec-69b23782cecb'
        url = f'https://api.sportradar.com/nba/trial/v8/en/games/{game_id}/pbp.json?api_key={api_key}'
        try:
            resp = requests.get(url, timeout=30)
            return resp.json()
        except Exception as exc:
            pytest.skip(f'Live API request failed: {exc}')
    if os.path.exists(fixture):
        with open(fixture, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    pytest.skip('No fixture available and RUN_LIVE_TESTS not enabled')


def test_first_quarter_scoring_and_rebounds():
    data = _load_data()
    assert isinstance(data, dict)
    assert 'periods' in data and data['periods'], 'Missing periods in PBP data'

    q1 = data['periods'][0]
    scoring_events = [e for e in q1.get('events', []) if e.get('event_type') in ['twopointmade', 'threepointmade', 'freethrowmade']]
    assert isinstance(scoring_events, list)

    # Print a few scoring plays for diagnostics
    for event in scoring_events[:10]:
        # basic shape checks
        assert 'description' in event
        assert 'clock' in event
        # if statistics present ensure it's list or dict
        if 'statistics' in event:
            assert isinstance(event['statistics'], (list, dict))

    # Rebounds
    rebound_events = [e for e in q1.get('events', []) if 'rebound' in e.get('event_type', '').lower()]
    for event in rebound_events[:5]:
        assert 'description' in event
        if 'statistics' in event:
            assert isinstance(event['statistics'], (list, dict))


