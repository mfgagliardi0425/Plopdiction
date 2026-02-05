import requests
from bs4 import BeautifulSoup

game_id = "401810365"
url = f"https://www.espn.com/nba/game/_/gameId/{game_id}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

title = soup.find('title')
if title:
    print(f"Title: {repr(title.string)}")
    title_text = title.string
    if ' vs ' in title_text:
        print("Found ' vs ' in title")
        parts = title_text.split(' vs ')
        print(f"Parts: {parts}")
    elif '-' in title_text:
        print("Found '-' in title")
        parts = title_text.split('-')
        print(f"Parts: {parts}")
