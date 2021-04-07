import sqlite3
import pandas as pd
import re


locstr = '../'


def clean(text):
    text = re.sub(r'\([^)]*\)', '', text.lower())
    text = re.sub(r'[^a-z0-9]', '', text)
    text = re.sub(r'(feat*)', '', text)
    return text


conn = sqlite3.connect(locstr + 'spoticry.db')
c = conn.cursor()

c.execute('PRAGMA foreign_keys = ON')
c.execute('DROP TABLE IF EXISTS listening;')
c.execute('DROP TABLE IF EXISTS songs;')
c.execute('DROP TABLE IF EXISTS listening_songs;')

make_listening = '''
CREATE TABLE listening (
  user_id VARCHAR(40) NOT NULL,
  song_id VARCHAR(18) NOT NULL,
  play_ct INTEGER NOT NULL,
  PRIMARY KEY (user_id, song_id)
);
'''

listening_df = pd.read_table(locstr + 'kaggle_visible_evaluation_triplets.txt',
                             names=['user_id', 'song_id', 'play_ct'], sep='\t')
listening_df.to_sql(name='listening', con=conn,
                    schema=make_listening, index=False)

make_songs = '''
CREATE TABLE songs (
  track_id VARCHAR(18) NOT NULL,
  song_id VARCHAR(18) NOT NULL, 
  artist VARCHAR(255) NOT NULL,
  title VARCHAR(255) NOT NULL,
  cleaned_title VARCHAR(255) NOT NULL,
  cleaned_artist VARCHAR(255) NOT NULL,
  PRIMARY KEY (song_id)
);
'''

songs_df = pd.read_csv(locstr + 'unique_tracks.txt', sep='\<SEP\>',
                       names=['track_id', 'song_id', 'artist', 'title'], engine='python')
songs_df.dropna(inplace=True)
songs_df['cleaned_title'] = songs_df.apply(lambda x: clean(x['title']), axis=1)
songs_df['cleaned_artist'] = songs_df.apply(
    lambda x: clean(x['artist']), axis=1)
songs_df.to_sql(name='songs', con=conn, schema=make_songs, index=False)

remove_dup = '''
DELETE FROM songs
WHERE rowid NOT IN
(SELECT MIN(rowid)
  FROM songs
  GROUP BY song_id);
'''
c.execute(remove_dup)
conn.commit()

make_listening_songs = '''
CREATE TABLE listening_songs (
  user_id VARCHAR(40) NOT NULL,
  song_id VARCHAR(18) NOT NULL, 
  artist VARCHAR(255) NOT NULL,
  title VARCHAR(255) NOT NULL,
  play_ct INTEGER NOT NULL,
  cleaned_title VARCHAR(255) NOT NULL,
  cleaned_artist VARCHAR(255) NOT NULL,
  PRIMARY KEY (user_id, song_id)
);
'''
c.execute(make_listening_songs)

join_tables = '''
INSERT INTO listening_songs (user_id, song_id, artist, title, play_ct, cleaned_title, cleaned_artist)
SELECT a.user_id, a.song_id, b.artist, b.title, a.play_ct, b.cleaned_title, b.cleaned_artist
FROM listening a
JOIN songs b
ON a.song_id = b.song_id
'''
c.execute(join_tables)
conn.commit()
conn.close()
