import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn


class SiameseLSTM(object):
    def BiRNN(self, x, x_lens, n_steps, n_hidden, biRnnScopeName, dropoutName):
        x = tf.nn.dropout(x, 0.5, name=dropoutName)
        # Forward direction cell
        lstm_fw_cell = rnn.LSTMCell(n_hidden)
        # Backward direction cell
        lstm_bw_cell = rnn.LSTMCell(n_hidden)
        # need scope to identified different cell
        outputs, output_states = nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length=x_lens,
                                                              dtype=tf.float32, scope=biRnnScopeName)
        return outputs, output_states

    def __init__(self, max_length, init_embeddings, num_classes, hidden_units,
                 embeddings_trainable=False, l2_reg_lambda=0):  # vocab_size, embedding_size
        with tf.name_scope('input'):
            u_shape = [None, max_length]  # self.batch_size
            v_shape = [None, num_classes, max_length]  # train:100  test:300
            self.input_x_u = tf.placeholder(tf.int32, u_shape, name="input_x_u")
            self.input_x_r = tf.placeholder(tf.int32, v_shape, name="input_x_r")
            self.input_y = tf.placeholder(tf.int64, [None], name="input_y")  # input_y: batch_size,
            self.u_lens = tf.placeholder(tf.int32, [None])
            self.v_lens = tf.placeholder(tf.int32, [None])
            self.n_steps = max_length
            self.hidden_size = hidden_units
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.embedding_size = np.shape(init_embeddings)[1]
            l2_loss = tf.constant(0.0, name="l2_loss")  # optional: l2 regularization loss

        with tf.name_scope("embedding"):
            # self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=embeddings_trainable,
            #                      name="W")
            self.W = tf.Variable(init_embeddings, trainable=embeddings_trainable, dtype=tf.float32, name='W')
            self.embedded_u = tf.nn.embedding_lookup(self.W, self.input_x_u)  # (batch_size, max_len, dim)
            print("DEBUG: embedded_u -> %s" % self.embedded_u)
            self.embedded_v = tf.nn.embedding_lookup(self.W, self.input_x_r)  # (batch_size, num_classes, max_len, dim)
            print ("DEBUG: embedded_v -> %s" % self.embedded_v)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.output1, _ = self.BiRNN(self.embedded_u, self.u_lens, self.n_steps, self.hidden_size,
                                         "relation_1", "relation_dropout_1")  # batch_size * dim
            outputs_fw, outputs_bw = self.output1  # ?, max_time, dim
            self.question_embedding = tf.concat([outputs_fw[:, -1, :], outputs_bw[:, 0, :]], 1)
            self.question_embedding = tf.expand_dims(self.question_embedding, 1)
            print ("DEBUG: question_embedding -> %s" % self.question_embedding ) # ?, 2*dim

            outputs_v_classes = []
            for j in range(num_classes):
                embedded_v = self.embedded_v[:, j, :, :]
                self.output2, _ = self.BiRNN(embedded_v, self.v_lens, self.n_steps, self.hidden_size,
                                             "relation_2_%d" % j,
                                             "relation_dropout_2")
                outputs_fw, outputs_bw = self.output2
                relation_embedding = tf.concat([outputs_fw[:, -1, :], outputs_bw[:, 0, :]], 1)
                # print "DEBUG: relation_embedding_temp -> %s" % relation_embedding
                relation_embedding_expand = tf.expand_dims(relation_embedding, 1)
                outputs_v_classes.append(relation_embedding_expand)
            self.relation_embedding = tf.concat(outputs_v_classes, 1)
            print ("DEBUG: relation_embedding -> %s" % self.relation_embedding)

        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            self.dot = tf.reduce_sum(tf.multiply(self.question_embedding, self.relation_embedding), 2)
            print ("DEBUG: dot -> %s" % self.dot)
            self.sqrt_u = tf.sqrt(tf.reduce_sum(self.question_embedding ** 2, 2))
            print ("DEBUG: sqrt_u -> %s" % self.sqrt_u)
            self.sqrt_r = tf.sqrt(tf.reduce_sum(self.relation_embedding ** 2, 2))
            print ("DEBUG: sqrt_r -> %s" % self.sqrt_r)
            epsilon = 1e-5
            self.cosine = tf.maximum(self.dot / (tf.maximum(self.sqrt_u * self.sqrt_r, epsilon)), epsilon)
            print ("DEBUG: cosine -> %s" % self.cosine)
            self.score = tf.nn.softmax(self.cosine)  # TODO
            print ("DEBUG: score -> %s" % self.score)
            self.predictions = tf.argmax(self.cosine, 1, name="predictions")
            print ("DEBUG: predictions -> %s" % self.predictions)

        # softmax regression - loss and prediction
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=100 * self.cosine)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="correct_num")
