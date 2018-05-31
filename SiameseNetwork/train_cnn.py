import os

import config
import math
import numpy as np
import sys
import tensorflow as tf
import time
from qa_cnn import QaCNN
from tqdm import tqdm
from util import util
from util.dataset import Dataset


def pull_batch(question_list, relation_list, label_list, batch_idx):
    batch_size = config.batch_size
    if (batch_idx + 1) * batch_size < len(question_list):
        question_list = question_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        relation_list = relation_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        label_list = label_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    else:  # last batch
        question_list = question_list[batch_idx * batch_size:]
        relation_list = relation_list[batch_idx * batch_size:]
        label_list = label_list[batch_idx * batch_size:]
    return question_list, relation_list, label_list


def train(train_set, dev_set, max_len, U):
    net = QaCNN(sequence_length=max_len,
                num_classes=config.neg_sample,
                filter_sizes=config.filter_sizes,
                num_filters=config.num_filters,
                init_embeddings=U,
                embeddings_trainable=config.embeddings_trainable,
                l2_reg_lambda=config.l2_reg_lambda)
    global_step = tf.Variable(0, name="global_step", trainable=True)
    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(net.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", net.loss)
        acc_summary = tf.summary.scalar("accuracy", net.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        # train_summary_dir = os.path.join(out_dir, "summaries", "train")
        # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        best_dev_loss, stop_step = sys.maxint, 0
        for epoch_idx in range(config.epoch_num):
            start = time.time()
            train_loss, dev_loss, train_correct_num, dev_correct_num = 0, 0, 0, 0

            for train_step in tqdm(range(int(math.ceil(train_set.size * 1.0 / config.batch_size))),
                                   desc='Training epoch ' + str(epoch_idx + 1) + ''):
                ques_batch, rela_batch, label_batch = pull_batch(train_set.ques_idx, train_set.rela_idx,
                                                                 train_set.label, train_step)
                feed_dict = {
                    net.input_x_u: ques_batch,
                    net.input_x_r: rela_batch,
                    net.input_y: label_batch,
                }
                _, summaries, score, loss, accuracy, correct_num, prediction, real_label = sess.run(
                    [train_op, train_summary_op, net.score, net.loss, net.accuracy, net.correct_num, net.predictions,
                     net.input_y], feed_dict)
                train_loss += loss
                train_correct_num += correct_num
                if train_step == 0:
                    train_score = score  # score:[0, 2]
                else:
                    train_score = np.concatenate((train_score, score), axis=0)

            for dev_step in tqdm(range(int(math.ceil(dev_set.size * 1.0 / config.batch_size))),
                                 desc='Deving epoch ' + str(epoch_idx + 1) + ''):
                ques_batch, rela_batch, label_batch = pull_batch(dev_set.ques_idx, dev_set.rela_idx,
                                                                 dev_set.label, dev_step)
                feed_dict = {
                    net.input_x_u: ques_batch,
                    net.input_x_r: rela_batch,
                    net.input_y: label_batch,
                }
                loss, accuracy, correct_num, score, summaries = sess.run(
                    [net.loss, net.accuracy, net.correct_num, net.score, dev_summary_op], feed_dict)
                dev_loss += loss
                dev_correct_num += correct_num
                if dev_step == 0:
                    dev_score = score
                else:
                    dev_score = np.concatenate((dev_score, score), axis=0)
            end = time.time()
            print(
                "epoch {}, time {}, train loss {:g}, train acc {:g},  dev loss {:g}, dev acc {:g}".format(
                    epoch_idx, end - start, train_loss / train_set.size, train_correct_num / train_set.size,
                               dev_loss / dev_set.size, dev_correct_num / dev_set.size))

            if dev_loss < best_dev_loss:
                stop_step = 0
                best_dev_loss = dev_loss
                print('saving new best result...')
                # print np.array(dev_set.rela).shape, dev_score.shape

                saver_path = saver.save(sess, "%s.ckpt" % checkpoint_prefix)
                print(saver_path)

                util.save_data(config.train_file, config.train_output_file, train_score.tolist(), train_set.rela)
                util.save_data(config.dev_file, config.dev_output_file, dev_score.tolist(), dev_set.rela)
            else:
                stop_step += 1
            if stop_step >= config.early_step:
                print('early stopping')
                break


def test(test_set, max_len, U):
    net = QaCNN(sequence_length=max_len,
                num_classes=config.num_classes,
                filter_sizes=config.filter_sizes,
                num_filters=config.num_filters,
                init_embeddings=U,
                embeddings_trainable=config.embeddings_trainable,
                l2_reg_lambda=config.l2_reg_lambda)
    # saver = tf.train.import_meta_graph("save/model.ckpt.meta")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, tf.train.latest_checkpoint("checkpoints/"))
        test_loss, test_correct_num = 0, 0
        start = time.time()
        for test_step in tqdm(range(int(math.ceil(test_set.size * 1.0 / config.batch_size))),
                              desc='Testing epoch ' + ''):
            ques_batch, rela_batch, label_batch = pull_batch(test_set.ques_idx, test_set.rela_idx,
                                                             test_set.label, test_step)
            feed_dict = {
                net.input_x_u: ques_batch,
                net.input_x_r: rela_batch,
                net.input_y: label_batch,
            }
            loss, accuracy, correct_num, score = sess.run(
                [net.loss, net.accuracy, net.correct_num, net.score], feed_dict)
            test_loss += loss
            test_correct_num += correct_num
            if test_step == 0:
                test_score = score
            else:
                test_score = np.concatenate((test_score, score), axis=0)
        util.save_data(config.test_file, config.test_output_file, test_score.tolist(), test_set.rela)
        end = time.time()
        print("time {}, test loss {:g}, train acc {:g}".format(end - start, test_loss / test_set.size,
                                                               test_correct_num / test_set.size))


if __name__ == "__main__":
    start = time.time()

    print("loading word embedding...")
    word_dict, embedding = util.get_pretrained_word_vector(config.word2vec_file, (config.voc_size, config.emb_size))
    print("vocabulary size: %d" % len(word_dict))

    print("loading train data...")
    x_u, x_r, y, _ = util.load_data(config.train_file, True, config.neg_sample)
    train_dataset = Dataset(x_u, x_r, y, config.max_sent_len, word_dict)
    print(np.array(train_dataset.ques_idx).shape, np.array(train_dataset.rela_idx).shape, np.array(
        train_dataset.label).shape)

    print("loading dev data...")
    x_u, x_r, y, _ = util.load_data(config.dev_file, True, config.neg_sample)
    dev_dataset = Dataset(x_u, x_r, y, config.max_sent_len, word_dict)
    print(np.array(dev_dataset.ques_idx).shape, np.array(dev_dataset.rela_idx).shape, np.array(dev_dataset.label).shape)

    print("loading test data...")
    x_u, x_r, y, _ = util.load_data(config.test_file, False, config.num_classes)
    test_dataset = Dataset(x_u, x_r, y, config.max_sent_len, word_dict)
    print(np.array(test_dataset.ques_idx).shape, np.array(test_dataset.rela_idx).shape, np.array(
        test_dataset.label).shape)

    print("training...")
    train(train_dataset, dev_dataset, config.max_sent_len, embedding)

    print("testing...")
    test(test_dataset, config.max_sent_len, embedding)

    end = time.time()
    print('total time: %s' % str(end - start))
