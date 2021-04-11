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
            length_dict[len(song_list)] += 1
            while len(song_list) >= window_size:
                added = True
                for i in range(window_size-1):
                    lab = song_list[i]
                    inp = song_list[:i] + song_list[i+1:window_size]
                    inputs.append(inp)
                    labels.append(lab)
                labels.append(song_list[window_size-1])
                inputs.append(song_list[:window_size-1])
                song_list = song_list[window_size:]
            song_list = []
            prev_user_id = song[1]

        id = song[0]
        vec = list(array.array('f', song[2]))
        if not id_dict.get(id)!= None:
            id_dict[id] = next_id
            embedding_table.append(vec)
            next_id += 1
        song_list.append(id_dict[id])


    # train_test split
    inputs = tf.convert_to_tensor(inputs)
    labels = tf.convert_to_tensor(labels)

    seed = np.random.randint(1,1000)
    inputs = tf.random.shuffle(inputs, seed)
    labels = tf.random.shuffle(labels, seed)

    # HYPER-PARAM TEST PERCENTAGE
    test_percentage = .2
    split_point = int(test_percentage * labels.shape[0])

    train_inputs = inputs[split_point:].numpy()
    train_labels = labels[split_point:].numpy()

    test_inputs = inputs[:split_point].numpy()
    test_labels = labels[:split_point].numpy()

    embedding_table = np.array(embedding_table)

    # saving data and embedding_table
    save_str = '/mnt/datassd/csci1951a-spoticry-data/transformer_data/'

    np.save(save_str + 'train_inputs', train_inputs)
    np.save(save_str + 'train_labels', train_labels)

    np.save(save_str + 'test_inputs', test_inputs)
    np.save(save_str + 'test_labels', test_labels)

    np.save(save_str + 'embedding_table', embedding_table)
    pass


if __name__ == "__main__":

    locstr = '/mnt/datassd2/spoticry-data/'

    see_tables = "SELECT name FROM sqlite_master WHERE type='table'"

    get_data = '''
    SELECT song_id, user_id, vec
    FROM transformer
    ORDER BY user_id
    '''

    conn = sqlite3.connect(locstr + 'spoticry.db')
    c = conn.cursor()

    c.execute(see_tables)
    print(c.fetchall())
    # c.execute(get_data)
    # data = c.fetchall()

    conn.close()

    # preprocess(data)


