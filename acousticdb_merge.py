#!/usr/bin/env python
# coding: utf-8

import sqlite3
import array
import glob

locstr = '/mnt/datassd2/csci1951a-spoticry-data/'


def add_sentiment(blob, sentiment_score):
    ar = array.array("f", blob)
    ar.append(sentiment_score)
    return ar


with sqlite3.connect(locstr + 'spoticry.db') as conn:
    conn.create_function("add_sentiment", 2, add_sentiment)
    c = conn.cursor()

    # Compile a tempory full acoustic database table from components
    make_main_acoustic = '''
  CREATE TABLE full_acoustic_brainz (
    music_brainz_recording_id VARCHAR NOT NULL,
    artist VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    cleaned_artist VARCHAR NOT NULL,
    cleaned_title VARCHAR NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (music_brainz_recording_id)
  );
  '''
    c.execute(make_main_acoustic)

    for file in glob.iglob(locstr + "acousticdb_components/*_acoustic.db"):
        c.execute('ATTACH DATABASE \'' + file + '\' AS \'component\'')
        c.execute('''
          INSERT INTO full_acoustic_brainz
          SELECT * FROM component.acoustic_brainz;
      ''')
        conn.commit()
        c.execute('DETACH DATABASE \'component\'')

# Remove duplicate songs from full table
    remove_dup = '''
      DELETE FROM full_acoustic_brainz
      WHERE rowid NOT IN
      (SELECT MIN(rowid)
        FROM full_acoustic_brainz
        GROUP BY cleaned_title, cleaned_artist);
    '''
    c.execute(remove_dup)

    # JOIN on spoticry data
    make_transformer_table = '''
      CREATE TABLE transformer (
        song_id VARCHAR(18) NOT NULL,
        user_id VARCHAR(40) NOT NULL,
        vec BLOB NOT NULL
      );
    '''
    c.execute(make_transformer_table)

    merge = '''
      INSERT INTO transformer
      SELECT listening_songs.song_id, user_id, add_sentiment(vec, sentiment)
      FROM full_acoustic_brainz a
      JOIN listening_songs b
      ON b.cleaned_title = a.cleaned_title AND b.cleaned_artist = a.cleaned_artist
      JOIN Sentiments c
      ON b.song_id = c.song_id;
    '''
    c.execute(merge)

    # Delete tempory table and close connection
    c.execute('DROP TABLE full_acoustic_brainz')
    conn.close()
