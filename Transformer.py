#!/usr/bin/env python
# coding: utf-8

# In[ ]:

window_size = 40 #????
embedding_size = 50
kqv_size = 64 

#Transformer

import tensorflow as tf

embedding_dict = {}

#loop through all matches in db and add their vectors as tensors to the dict and make them untrainable


W_q = tf.Variable(tf.random.normal(shape = (embedding_size, kqv_size)))
W_k = tf.Variable(tf.random.normal(shape = (embedding_size, kqv_size)))
W_v = tf.Variable(tf.random.normal(shape = (embedding_size, kqv_size)))



# making training data
#remove 1 song from sequence of songs and hold that out as a label for each
#also could mask a certain percentage like BERT

