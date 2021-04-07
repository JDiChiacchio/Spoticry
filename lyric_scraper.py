from bs4 import BeautifulSoup
import requests
import sqlite3
import re
import concurrent.futures
import time
import os
from humanfriendly import format_timespan


# Database path
dbpath = '../spoticry.db'
# True to delete lyric table and start from scratch. False to continue on unscraped song_ids
rebuild = False
# Execute this many requests per database update/progress report
block_size = 5000


def format_request(song_tuple):
    artist, title = song_tuple[1:]
    Source_URL = "https://genius.com/"
    query = artist + '-' + title + '-lyrics'
    # Convert spaces to hyphens
    query = re.sub(r'\s+', '-', query.lower())
    # Restrict query chars to alphanumeric and hyphens
    query = re.sub(r'[^a-z0-9\-]', '', query)
    return (song_tuple[0], Source_URL + query)


def save_and_format(response, curs):
    ret = 0
    if response[1].status_code == 200:
        soup = BeautifulSoup(response[1].text, 'html.parser')
        try:
            text = soup.body.find("div", "lyrics", recursive=True).get_text()
            # reduce all whitespace to single spaces
            text = " ".join(text.split())
        except AttributeError:
            pass
        else:
            curs.execute('INSERT INTO lyrics VALUES (?, ?)',
                         (response[0], text))
            ret = 1
    return ret


def request_worker(request):
    response = requests.get(request[1])
    return (request[0], response)


conn = sqlite3.connect('../spoticry.db')
c = conn.cursor()
if rebuild:
    c.execute('DROP TABLE IF EXISTS lyrics;')
    c.commit()
    c.execute('''
        CREATE TABLE lyrics(
        song_id VARCHAR(18) NOT NULL, 
        lyrics VARCHAR(10000),
        PRIMARY KEY (song_id)
        );
    ''')
    conn.commit()
c.execute('''
    SELECT DISTINCT songs.song_id, artist, title
    FROM songs
    JOIN listening ON songs.song_id = listening.song_id
    WHERE songs.song_id NOT IN (
        SELECT song_id
        FROM lyrics);
 ''')
song_urls = [format_request(song_entry) for song_entry in c.fetchall()]
conn.close()

num_titles = len(song_urls)
suc_sum = 0
base_time = time.time()
# Use a thread pool to execute multiple requests at once
for i in range(round(num_titles / block_size)):
    upper_bound = min(block_size*(i+1), num_titles - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        html_out = list(executor.map(
            request_worker, song_urls[block_size * i: upper_bound]))
        conn = sqlite3.connect(dbpath)
        c = conn.cursor()
        suc = [save_and_format(item, c) for item in html_out]
        conn.commit()
        conn.close()
    progress = round(100 * upper_bound/num_titles)
    time_spent = time.time() - base_time
    projected = (time_spent / upper_bound) * (num_titles - upper_bound)
    suc_sum += sum(suc)
    suc_rate = round(100 * suc_sum/upper_bound)
    if i % 10 == 0:
        os.system("clear")
    print(f'Progress: {progress}%  ({upper_bound}/{num_titles} songs checked) | Success rate: {suc_rate}% \nTime elapsed: {format_timespan(time_spent)}. | Projected time remaining: {format_timespan(projected)}.')
