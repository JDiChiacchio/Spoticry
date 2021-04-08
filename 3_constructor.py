#!/usr/bin/env python
# coding: utf-8

import sqlite3
import glob
import re
import array
import json
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import layers, models


locstr = '/mnt/datassd/'
RETRAIN = False

# Acousticbrainz feature extraction and database creating code


conn_acoust = sqlite3.connect('3_acoustic.db')
acoust = conn_acoust.cursor()


acoust.execute('PRAGMA foreign_keys = ON')
acoust.execute('PRAGMA journal_mode = OFF')
acoust.execute('DROP TABLE IF EXISTS acoustic_brainz;')

make_acoustic = '''
CREATE TABLE acoustic_brainz (

  music_brainz_recording_id VARCHAR NOT NULL,
  artist VARCHAR NOT NULL,
  title VARCHAR NOT NULL,
  cleaned_artist VARCHAR NOT NULL,
  cleaned_title VARCHAR NOT NULL,
 
  vec BLOB NOT NULL,

  PRIMARY KEY (music_brainz_recording_id)

);
'''
acoust.execute(make_acoustic)

conn_acoust.commit()


class Autoencoder(models.Model):
    def __init__(self, original_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.encoder = models.Sequential([
            layers.Dense(latent_dim, activation='linear'),
        ])
        self.decoder = models.Sequential([
            layers.Dense(original_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def key_to_dummies(key):
    notes = {
        'A': 0, 'A#': 1, 'Ab': 2, 'B': 3, 'B#': 4, 'Bb': 5, 'C': 6,
        'C#': 7, 'Cb': 8, 'D': 9, 'D#': 10, 'Db': 11, 'E': 12, 'E#': 13,
        'Eb': 14, 'F': 15, 'F#': 16, 'Fb': 17, 'G': 18, 'G#': 19, 'Gb': 20
    }
    vector = [0 for _ in notes.keys()]
    mark_ind = notes.get(key)
    if mark_ind is not None:
        vector[mark_ind] = 1
        return vector
    else:
        raise Exception('Unrecognized string: ' + key)


def json_to_vector(j):
    vec = []
    id = j['metadata']['tags']['musicbrainz_recordingid'][0]
    title = j['metadata']['tags']['title'][0]
    artist = j['metadata']['tags']['artist'][0]
    for k, v in j.items():
        if (k == 'metadata'):
            continue
        for key in v.keys():
            val = j[k][key]
            if isinstance(val, list):
                if not key == 'beats_position':
                    vec.extend(val)
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, list):
                        if isinstance(v2[0], list):
                            vec.extend([i for sub in v2 for i in sub])
                        else:
                            vec.extend(v2)
                    elif isinstance(v2, int) or isinstance(v2, float):
                        vec.append(v2)
                    else:
                        raise Exception('Unrecognized value: ' + str(v))
            elif isinstance(val, str):
                if (val == 'major'):
                    vec.append(1)
                elif (val == 'minor'):
                    vec.append(0)
                else:
                    vec.extend(key_to_dummies(val))
            elif isinstance(val, int) or isinstance(val, float):
                vec.append(val)
            else:
                raise Exception('Unrecognized value: ' + str(val))
    return id, title, artist, vec


def get_vectors(num_vectors):
    vecs = []
    songs_info = []
    num_times = 0
    for path in glob.glob(locstr + "./acoustic_brainz_dataset/*/*.json"):
        if len(vecs) > num_vectors:
            break
        num_times += 1

        if not num_times % 500:
            print(num_times, "len(vecs):", len(vecs), path)

        j = json.load(open(path))
        if j['metadata']['tags'].get('title') and j['metadata']['tags'].get('artist'):

            id, title, artist, vec = json_to_vector(j)
            songs_info.append((id, title, artist))
            vecs.append(vec)

    return songs_info, np.array(vecs)


def normalize(vecs, mins, maxs):
    # print("normalizing")
    return np.nan_to_num((vecs - mins) / (maxs - mins)).astype('float32')


def clean(text):
    text = re.sub(r'\([^)]*\)', '', text.lower())
    text = re.sub(r'[^a-z0-9]', '', text)
    text = re.sub(r'(feat*)', '', text)
    return text


def add_to_db(id, title, artist, vec):
    to_db = [id, artist, title, clean(artist), clean(title)]
    to_db.append(array.array('f', vec))

    acoust.execute(
        'INSERT INTO acoustic_brainz VALUES (?, ?, ?, ?, ?, ?)', to_db)


def train_autoencoder():
    songs_info, vecs = get_vectors(5000)
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    vecs = normalize(vecs, mins, maxs)

    ae = Autoencoder(2691, 50)
    ae.compile(optimizer='adam', loss=MeanSquaredError())
    ae.fit(vecs, vecs, epochs=10, shuffle=True)
    ae.summary()
    ae.save(locstr + 'ae-model')
    np.save(locstr + 'mins.npy', mins)
    np.save(locstr + 'maxs.npy', maxs)
    return ae, mins, maxs


def load_autoencoder():
    ae = models.load_model(locstr + 'ae-model')
    mins = np.load(locstr + 'mins.npy')
    maxs = np.load(locstr + 'maxs.npy')
    return ae, mins, maxs


if RETRAIN:
    ae, mins, maxs = train_autoencoder()
else:
    ae, mins, maxs = load_autoencoder()


num_times = 0
of_dif_length = 0
no_title_or_artist = 0
added = 0
seen = set()
for path in glob.iglob(locstr + "acoustic_brainz_dataset/3*/*.json"):
    # print(path)
    num_times += 1

    if not num_times % 500:
        print(num_times, path, of_dif_length)

    j = json.load(open(path))
    if j['metadata']['tags'].get('title') and j['metadata']['tags'].get('artist') and j['metadata']['tags'].get('musicbrainz_recordingid')[0] not in seen:
        id, title, artist, vec = json_to_vector(j)
        seen.add(id)
        if len(vec) == 2691:
            add_to_db(id, title, artist, ae.call(
                np.array([normalize(vec, mins, maxs)]))[0].numpy())
            added += 1
        else:
            of_dif_length += 1
    else:
        no_title_or_artist += 1

conn_acoust.commit()
conn_acoust.close()
print("done", of_dif_length, no_title_or_artist, added)
