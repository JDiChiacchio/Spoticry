#!/usr/bin/env python
# coding: utf-8

import sqlite3
import glob

locstr = '/mnt/datassd/csci1951a-spoticry-data/'


conn = sqlite3.connect(locstr + 'acoustic.db')
c = conn.cursor()


c.execute('DROP TABLE IF EXISTS full_acoustic_brainz')
conn.commit()
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
conn.commit()
for file in glob.iglob(locstr + "acousticdb_components/*_acoustic.db"):
    c.execute('ATTACH DATABASE \'' + file + '\' AS \'component\'')
    c.execute('''
        INSERT INTO full_acoustic_brainz
        SELECT * FROM component.acoustic_brainz;
    ''')
    conn.commit()
    c.execute('DETACH DATABASE \'component\'')
conn.close()
