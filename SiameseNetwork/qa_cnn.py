import tensorflow as tf
import numpy as np
import config


class QaCNN(object):
    def __init__(self, sequence_length, num_classes, filter_sizes,
                 num_filters, init_embeddings,
                 embeddings_trainable=False, l2_reg_lambda=0.0):
        self.batch_size = config.batch_size
        self.embedding_size = np.shape(init_embeddings)[1]
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # self.train_flag = tf.placeholder(dtype=tf.bool, name='bool')
        with tf.name_scope('input'):
            u_shape = [None, sequence_length]  # self.batch_size
            v_shape = [None, num_classes, sequence_length]  # train:100  test:300
            self.input_x_u = tf.placeholder(tf.int32, u_shape, name="input_x_u")
            self.input_x_r = tf.placeholder(tf.int32, v_shape, name="input_x_r")
            self.input_y = tf.placeholder(tf.int64, [None], name="input_y")  # input_y: batch_size,

        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(init_embeddings, trainable=embeddings_trainable, dtype=tf.float32, name='W')
            self.embedded_u = tf.nn.embedding_lookup(W, self.input_x_u)  # bs x seq_len x emb_size
            print ("DEBUG: embedded_u -> %s" % self.embedded_u)
            self.embedded_r = tf.nn.embedding_lookup(W, self.input_x_r)  # bs x neg_size x seq_len x emb_size
            print ("DEBUG: embedded_r -> %s" % self.embedded_r)
            self.embedded_u_expanded = tf.expand_dims(self.embedded_u, -1)  # bs x seq_len x emb_size x 1
            print ("DEBUG: embedded_u_expanded -> %s" % self.embedded_u_expanded)
            self.embedded_r_expanded = tf.expand_dims(self.embedded_r, -1)  # bs x neg_size x seq_len x emb_size x 1
            print ("DEBUG: embedded_r_expanded -> %s" % self.embedded_r_expanded)

        # Create a convolution + maxpooling layer for each filter size
        pooled_outputs_u = []
        pooled_outputs_r = []
        for i, filter_size in enumerate(filter_sizes):  # [2, 3, 4]
            with tf.name_scope("conv-maxpool-%s" % filter_size):  # 50
                # Convolution layer
                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                conv_u = tf.nn.conv2d(
                    self.embedded_u_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-u")

                # Apply nonlinearity
                h_u = tf.nn.sigmoid(tf.nn.bias_add(conv_u, b), name="activation-u")

                # Maxpooling over outputs
                pooled_u = tf.nn.max_pool(
                    h_u,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool-u")
                pooled_outputs_u.append(pooled_u)  # 1 x num_filters

                # Pass each element in x_r through the same layer
                pooled_outputs_r_classes = []

                # num_classes = tf.where(self.train_flag, config.negative_size, config.max_candidate_relation)

                for j in range(num_classes):
                    embedded_r = self.embedded_r_expanded[:, j, :, :, :]
                    conv_r_j = tf.nn.conv2d(
                        embedded_r,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-r-%s" % j)

                    h_r_j = tf.nn.sigmoid(tf.nn.bias_add(conv_r_j, b), name="activation-r-%s" % j)

                    pooled_r_j = tf.nn.max_pool(
                        h_r_j,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool-r-%s" % j)
                    pooled_outputs_r_classes.append(pooled_r_j)
                    # print "DEBUG: pooled_outputs_r_classes -> %s" % pooled_outputs_r_classes

                # out_tensor: batch_size x 1 x num_class x num_filters
                out_tensor = tf.concat(pooled_outputs_r_classes, 2)
                print ("DEBUG: out_tensor -> %s" % out_tensor)
                pooled_outputs_r.append(out_tensor)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print ("DEBUG: pooled_outputs_u -> %s" % pooled_outputs_u)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        print ("DEBUG: h_pool_u -> %s" % self.h_pool_u)
        # batch_size x 1 x num_filters_total
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, 1, num_filters_total])
        print ("DEBUG: h_pool_flat_u -> %s" % self.h_pool_flat_u)

        print ("DEBUG: pooled_outputs_r -> %s" % pooled_outputs_r)
        self.h_pool_r = tf.concat(pooled_outputs_r, 3)
        print ("DEBUG: h_pool_r -> %s" % self.h_pool_r)
        # h_pool_flat_r: batch_size x num_classes X num_filters_total
        self.h_pool_flat_r = tf.reshape(self.h_pool_r, [-1, num_classes, num_filters_total])
        print ("DEBUG: h_pool_flat_r -> %s" % self.h_pool_flat_r)

        # # Add dropout layer to avoid overfitting
        # with tf.name_scope("dropout"):
        #     self.h_features = tf.concat([self.h_pool_flat_u, self.h_pool_flat_r], 1)
        #     print "DEBUG: h_features -> %s" % self.h_features
        #     self.h_features_dropped = tf.nn.dropout(self.h_features, config.dropout_keep_prob)
        #
        #     self.h_dropped_u = self.h_features_dropped[:, :1, :]
        #     self.h_dropped_r = self.h_features_dropped[:, 1:, :]
        #     print "DEBUG: h_dropped_u -> %s" % self.h_dropped_u
        #     print "DEBUG: h_dropped_r -> %s" % self.h_dropped_r

        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            self.dot = tf.reduce_sum(tf.multiply(self.h_pool_flat_u, self.h_pool_flat_r), 2)
            print ("DEBUG: dot -> %s" % self.dot)
            self.sqrt_u = tf.sqrt(tf.reduce_sum(self.h_pool_flat_u ** 2, 2))
            print ("DEBUG: sqrt_u -> %s" % self.sqrt_u)
            self.sqrt_r = tf.sqrt(tf.reduce_sum(self.h_pool_flat_r ** 2, 2))
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
