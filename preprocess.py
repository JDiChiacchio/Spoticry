import sqlite3
import array
import tensorflow as tf
import numpy as np
from collections import defaultdict

def preprocess(data):
    next_id = 0
    prev_user_id = "noonehasthisid"
    id_dict = {}
    embedding_table = []
    inputs = []
    labels = []
    song_list = []
    window_size = 4

    length_dict = defaultdict(int)

    #loop through data vectorizing and padding
    for song in data:
        if song[1] != prev_user_id:
            while len(song_list) >= 4:
                inputs.append(song_list[:window_size-1])
                labels.append(song_list[window_size-1])
                song_list = song_list[window_size:]
            length_dict[len(song_list)] += 1
            song_list = []
            prev_user_id = song[1]
        id = song[0]
        vec = list(array.array('f', song[2]))
        if not id_dict.get(id)!= None:
            id_dict[id] = next_id
            embedding_table.append(vec)
            next_id += 1
        song_list.append(id_dict[id])

    #train_test split
    inputs = tf.convert_to_tensor(inputs)
    labels = tf.convert_to_tensor(labels)

    seed = np.random.randint(1,1000)
    inputs = tf.random.shuffle(data, seed)
    labels = tf.random.shuffle(data, seed)

    # HYPER-PARAM TEST PERCENTAGE
    test_percentage = .2
    split_point = test_percentage * labels.shape[0]

    train_inputs = inputs[split_point:]
    train_labels = labels[split_point:]

    test_inputs = inputs[:split_point]
    test_labels = labels[:split_point]

    print(train_inputs.shape, train_labels.shape)
    print(test_inputs[0].numpy)

    # return train_inputs, train_labels, test_inputs, test_labels, embedding_table, id_dict
    pass


if __name__ == "__main__":

    locstr = '/mnt/datassd/csci1951a-spoticry-data/'

    see_tables = "SELECT name FROM sqlite_master WHERE type='table'"

    get_data = '''
    SELECT song_id, user_id, vec
    FROM transformer
    ORDER BY user_id
    '''

    conn = sqlite3.connect(locstr + 'acoustic.db')
    c = conn.cursor()


    c.execute(get_data)
    data = c.fetchall()

    conn.close()

    preprocess(data)


