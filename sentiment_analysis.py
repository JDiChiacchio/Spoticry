#!/usr/bin/env python
# coding: utf-8

import sqlite3
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PyDictionary import PyDictionary
nltk.download('vader_lexicon')
nltk.download('words')
nltk.download('stopwords')


locstr = '../data/'


dictionary = PyDictionary()
sid = SentimentIntensityAnalyzer()


conn = sqlite3.connect(locstr + 'spoticry.db')
c = conn.cursor()
c.execute("SELECT song_id, lyrics FROM lyrics")
# STEP 1: Clean the data some more + remove stopwords

valid_words = set(nltk.corpus.words.words())
sentiments = []
#stopwords: translation
for song in c.fetchall():
    # remove identifiers like chorus, verse, etc
    cleaned_lyrics = re.sub(r'[\(\[].*?[\)\]]', '', song[1])
    # word list should hold each word
    word_list = cleaned_lyrics.split()

    total_score = 0

    total_valid_words = 0

    num_english = 0
    num_non_english = 0

    for word in word_list:

        # not sure if it does return None if the word is not found, just a guess
        if word not in valid_words:
            num_non_english += 1
        else:
            num_english += 1

        if word not in valid_words or word in nltk.corpus.stopwords.words('english'):
            word_list.remove(word)
        else:
            # can go word by word or group by 10 words
            score = sid.polarity_scores(word)
            score = score["compound"]
            total_score += score
            total_valid_words += 1
    if total_valid_words != 0 and num_non_english < num_english:
        average_score = total_score / total_valid_words
        sentiments.append((song[0], average_score))

make_sentiments_table = '''
  CREATE TABLE Sentiments(
  song_id VARCHAR(18) NOT NULL, 
  sentiment real,
  PRIMARY KEY (song_id)
);
'''
c.execute("DROP TABLE IF EXISTS Sentiments")
conn.commit()
c.execute(make_sentiments_table)
conn.commit()

for song in sentiments:
    c.execute('INSERT INTO Sentiments VALUES (?, ?)', song)
conn.commit()
conn.close()
