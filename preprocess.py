import sqlite3
import array

def preprocess(data):
    next_id = 0
    id_dict = {} #music_brainz_id to num
    embedding_table = []
    inputs = []
    labels = []

    #loop through data vectorizing and padding
    for user in data:
        song_list = []
        for song in user:
            id = song[0]
            vec = song[2]
            if not id_dict.get(id)!= None:
                id_dict[id] = next_id
                embedding_table.append(vec)
                next_id += 1
            song_list.append(id_dict[id])
        

    #train_test split
    inputs = tf.tensor(inputs)
    labels = tf.tensor(labels)

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

    return train_inputs, train_labels, test_inputs, test_labels, embedding_table, id_dict


if __name__ == "__main__":

    locstr = '/mnt/datassd/csci1951a-spoticry-data/'

    see_tables = "SELECT name FROM sqlite_master WHERE type='table'"

    get_data = '''
    SELECT song_id, user_id, vec
    FROM transformer
    GROUP BY user_id
    LIMIT 1
    '''

    conn = sqlite3.connect(locstr + 'acoustic.db')
    c = conn.cursor()


    c.execute(get_data)
    temp = c.fetchall()
    print(temp[-1])
    vec = array.array('f', temp[-1])
    print(vec)
    print(len(vec))

    conn.close()

