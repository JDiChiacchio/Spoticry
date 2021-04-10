#!/usr/bin/env python
# coding: utf-8
from comet_ml import Experiment
import tensorflow as tf
from preprocess import *

hyper_params = {
     "batch_size": 16,
     "num_epochs": 1,
     "learning_rate": .001,
     "window_size": 3, #lol :(
     "embedding_size": 51,
     "kqv_size": 64
 }

class Transformer(tf.Module):

    def __init__(self, embedding_table):
        super().__init__()

        self.batch_size = hyper_params["batch_size"]
        self.num_epochs = hyper_params["num_epochs"]
        self.lr = hyper_params["learning_rate"]
        self.window_size = hyper_params["window_size"]
        self.embedding_size = hyper_params["embedding_size"]
        self.kqv_size = hyper_params["kqv_size"]

        self.W_k = tf.Variable(tf.random.normal(shape = (self.embedding_size, self.kqv_size)))
        self.W_q = tf.Variable(tf.random.normal(shape = (self.embedding_size, self.kqv_size)))
        self.W_v = tf.Variable(tf.random.normal(shape = (self.embedding_size, self.kqv_size)))

        self.dense = tf.Variable(tf.random.normal(shape = (self.kqv_size, self.embedding_size)))
        self.bias = tf.Variable(tf.random.normal(shape = (self.embedding_size,)))

        self.embeddings = tf.convert_to_tensor(embedding_table)

    def get_embedding(self, indices):
        return tf.gather(self.embeddings,indices)

    def forward(self, input):
        #takes in batches of size batch_size x window_size

        embedded = self.get_embedding(input)

        k = tf.linalg.matmul(embedded, self.W_k)
        q = tf.linalg.matmul(embedded, self.W_q)
        v = tf.linalg.matmul(embedded, self.W_v)

        z = tf.matmul(tf.nn.softmax(tf.matmul(q,k.T)/(self.kqv_size **(.5))), v)

        return tf.matmul(z, self.dense) + self.bias


def train(model, inputs, labels):

    optimizer = tf.keras.optimizers.Adam(model.lr)

    #shuffling
    seed = np.random.randint(1,1000)
    inputs = tf.random.shuffle(inputs, seed)
    labels = tf.random.shuffle(labels, seed)
    num_batches = inputs.shape[0]//model.batch_size

    for epoch in range(model.num_epochs):
        for batch in range(num_batches):
            with tf.GradientTape() as tape:

                batch_inputs = inputs[batch*model.batch_size:(batch+1)*model.batch_size]
                batch_labels = labels[batch*model.batch_size:(batch+1)*model.batch_size]

                out = model.forward(batch_inputs)
                loss = tf.keras.losses.MSE(model.get_embedding(batch_labels), out)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                experiment.log_metric("loss",loss,step= epoch*num_batches + batch)

def test(model, inputs, labels):

    loss_fn = tf.keras.losses.MSE()
    model = model.evaluate()

    total_model_loss = 0.0
    total_avg_loss = 0.0

    num_batches = inputs.shape[0]//model.batch_size
    for batch in range(num_batches):

        batch_inputs = inputs[batch*model.batch_size:(batch+1)*model.batch_size]
        batch_labels = labels[batch*model.batch_size:(batch+1)*model.batch_size]
        batch_labels = model.get_embedding(batch_labels)

        model_out = model.forward(batch_inputs)
        avg_out = tf.math.reduce_mean(batch_inputs, axis=1)

        model_loss = tf.keras.losses.MSE(batch_labels, model_out)
        avg_loss = loss_fn(batch_labels, avg_out)

        total_model_loss += model_loss
        total_avg_loss += avg_loss

    print("model loss:", total_model_loss)
    print("avg loss:", total_avg_loss)
    experiment.log_metric("model test loss", total_model_loss)
    experiment.log_metric("avg test loss", total_avg_loss)

if __name__ == "__main__":

    experiment = Experiment(
        api_key="2jaNx8WWN5smqSk7XGawlXOFF",
        project_name="spoticry",
        workspace="gvannewk",
    )

    experiment.log_parameters(hyper_params)

    locstr = '/mnt/datassd/csci1951a-spoticry-data/transformer_data/'

    train_inputs = tf.convert_to_tensor(np.load(locstr + 'train_inputs.npy'))
    train_labels = tf.convert_to_tensor(np.load(locstr + 'train_labels.npy'))
    test_inputs = tf.convert_to_tensor(np.load(locstr + 'test_inputs.npy'))
    test_labels = tf.convert_to_tensor(np.load(locstr + 'test_labels.npy'))
    embedding_table = np.load(locstr + 'embedding_table.npy')

    model = Transformer(embedding_table)
    train(model, train_inputs, train_labels)
    #save model
    tf.saved_model.save(model, locstr + 'transformer_model')

    test(model, test_inputs, test_labels)
