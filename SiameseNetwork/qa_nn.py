import tensorflow as tf
import numpy as np
import config


class QaNN(object):
    def __init__(self, sequence_length, init_embeddings, num_classes,
                 embeddings_trainable=False, l2_reg_lambda=0.0):
        self.L1 = 300
        self.L2 = 120
        self.batch_size = config.batch_size
        self.embedding_size = np.shape(init_embeddings)[1]
        # print self.embedding_size
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        with tf.name_scope('input'):
            u_shape = [None, sequence_length]  # self.batch_size
            v_shape = [None,  num_classes, sequence_length]  # train:100  test:300
            self.input_x_u = tf.placeholder(tf.int32, u_shape, name="input_x_u")
            self.input_x_r = tf.placeholder(tf.int32, v_shape, name="input_x_r")
            self.input_y = tf.placeholder(tf.int64, [None], name="input_y")  # input_y: batch_size,

        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(init_embeddings, trainable=embeddings_trainable, dtype=tf.float32, name='W')
            self.embedded_u = tf.nn.embedding_lookup(W, self.input_x_u)  # batch_size x sent_len x embedding_size
            self.embedded_u = tf.reduce_sum(self.embedded_u, 1)  # batch_size x embedding_size
            print ("DEBUG: embedded_u -> %s" % self.embedded_u)
            self.embedded_r = tf.nn.embedding_lookup(W, self.input_x_r)
            self.embedded_r = tf.reduce_sum(self.embedded_r, 2)  # batch_size x neg_size x embedding_size
            self.embedded_r = tf.reshape(self.embedded_r, [-1, self.embedding_size])
            print ("DEBUG: embedded_r -> %s" % self.embedded_r)

        with tf.name_scope('L1'):
            l1_par_range = np.sqrt(6.0 / (self.embedding_size + self.L1))
            weight1 = tf.Variable(tf.random_uniform([self.embedding_size, self.L1], -l1_par_range, l1_par_range))
            bias1 = tf.Variable(tf.random_uniform([self.L1], -l1_par_range, l1_par_range))
            query_l1 = tf.matmul(self.embedded_u, weight1) + bias1
            doc_l1 = tf.matmul(self.embedded_r, weight1) + bias1
            self.query_l1 = tf.nn.relu(query_l1)
            print("DEBUG: query_l1 -> %s" % self.query_l1)
            self.doc_l1 = tf.nn.relu(doc_l1)
            print ("DEBUG: doc_l1 -> %s" % self.doc_l1)

        with tf.name_scope('L2'):
            l2_par_range = np.sqrt(6.0 / (self.L1 + self.L2))
            weight2 = tf.Variable(tf.random_uniform([self.L1, self.L2], -l2_par_range, l2_par_range))
            bias2 = tf.Variable(tf.random_uniform([self.L2], -l2_par_range, l2_par_range))
            query_l2 = tf.matmul(self.query_l1, weight2) + bias2
            doc_l2 = tf.matmul(self.doc_l1, weight2) + bias2
            self.query_l2 = tf.nn.relu(query_l2)
            self.query_l2 = tf.expand_dims(self.query_l2, 1)
            print ("DEBUG: query_l2 -> %s" % self.query_l2)
            self.doc_l2 = tf.nn.relu(doc_l2)
            self.doc_l2 = tf.reshape(self.doc_l2, [-1, num_classes, self.L2])
            print ("DEBUG: doc_l2 -> %s" % self.doc_l2)
            # doc_y = tf.nn.dropout(doc_y, 0.5)

        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            self.dot = tf.reduce_sum(tf.multiply(self.query_l2, self.doc_l2), 2)
            print ("DEBUG: dot -> %s" % self.dot)
            self.sqrt_u = tf.sqrt(tf.reduce_sum(self.query_l2 ** 2, 2))
            print ("DEBUG: sqrt_u -> %s" % self.sqrt_u)
            self.sqrt_r = tf.sqrt(tf.reduce_sum(self.doc_l2 ** 2, 2))
            print ("DEBUG: sqrt_r -> %s" % self.sqrt_r)
            epsilon = 1e-5
            self.cosine = tf.maximum(self.dot / (tf.maximum(self.sqrt_u * self.sqrt_r, epsilon)), epsilon)
            print("DEBUG: cosine -> %s" % self.cosine)
            self.score = tf.nn.softmax(self.cosine)  # TODO: score:[0, 2]
            print("DEBUG: score -> %s" % self.score)
            self.predictions = tf.argmax(self.cosine, 1, name="predictions")
            print("DEBUG: predictions -> %s" % self.predictions)

        # softmax regression - loss and prediction
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=100 * self.cosine)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="correct_num")