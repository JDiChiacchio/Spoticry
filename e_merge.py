#!/usr/bin/env python
# coding: utf-8

# In[96]:


#Imports
import sqlite3

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
locstr = '/content/drive/Shareddrives/Spoticry/'


# In[97]:


c = sqlite3.connect(locstr + 'f_acoustic.db')
accoust = c.cursor()
accoust.execute('ATTACH DATABASE "/content/drive/Shareddrives/Spoticry/spoticry.db" AS spoticry')


# In[92]:


accoust.execute('SELECT cleaned_artist FROM acoustic_brainz')
print(accoust.fetchone())

c.close()


# In[ ]:


#Testing naming differences
e_title = '''
SELECT title
FROM acoustic_brainz
'''

accoust.execute(e_title)
for _ in range(500):
  print(accoust.fetchmany(7))

c.close()


# In[ ]:


spoticry_title = '''
SELECT title
FROM spoticry.listening_songs
'''

accoust.execute(spoticry_title)
for _ in range(500):
  print(accoust.fetchmany(7))

c.close()


# In[93]:


#Testing artist join to see differences in title naming conventions
join = '''
SELECT spoticry.listening_songs.title, acoustic_brainz.title
FROM acoustic_brainz JOIN spoticry.listening_songs ON
acoustic_brainz.artist = spoticry.listening_songs.artist and acoustic_brainz.title = spoticry.listening_songs.title
'''

accoust.execute(artist_join)
for _ in range(150):
  print(accoust.fetchone())

c.close()


# In[98]:


#Testing what percent of songs in listening are in accoustic

join_count = '''
  SELECT COUNT(DISTINCT spoticry.listening_songs.title)
  FROM acoustic_brainz JOIN spoticry.listening_songs ON
  acoustic_brainz.cleaned_title = spoticry.listening_songs.cleaned_title and acoustic_brainz.cleaned_artist = spoticry.listening_songs.cleaned_artist
'''

# artist_join_count = '''
# SELECT COUNT(DISTINCT spoticry.listening_songs.title)
# FROM acoustic_brainz JOIN spoticry.listening_songs ON
# acoustic_brainz.artist = spoticry.listening_songs.artist
# '''

# song_join= '''
# SELECT DISTINCT spoticry.listening_songs.title, spoticry.listening_songs.artist, acoustic_brainz.title
# FROM acoustic_brainz JOIN spoticry.listening_songs ON
# acoustic_brainz.artist = spoticry.listening_songs.artist
# '''

e_count = '''
SELECT COUNT(*)
FROM acoustic_brainz
'''

spoticry_count = '''
SELECT COUNT(*)
FROM spoticry.listening_songs
'''


accoust.execute(e_count)
c.commit()
num_e = accoust.fetchone()

accoust.execute(join_count)
c.commit()
num_joined = accoust.fetchone()
# for _ in range(50):
#   print(accoust.fetchmany(4))

accoust.execute(spoticry_count)
c.commit()
num_spoticry = accoust.fetchone()


print(num_joined, num_e, num_spoticry)
print(num_joined[0]/num_spoticry[0]*16*(16)*100, "% Projected")

c.close()

