import os
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict
import random
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_questions(filename):
    print("getting questions ...")
    id2questions = {}
    id2goldrelas = {}
    qids = list()
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        lineid = items[0].strip()
        # mid = items[1].strip()
        question = items[5].strip()
        rel = items[3].strip()
        qids.append(lineid)
        id2questions[lineid] = question
        id2goldrelas[lineid] = rel
    return qids, id2questions, id2goldrelas


# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits=100):
    print("Entity Source : {}".format(filename))
    id2mids = defaultdict(list)
    qids = list()
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0]
        cand_mids = items[1:]  # [:hits]
        qids.append(lineid)
        for mid_entry in cand_mids:
            mid, mid_name, mid_type, score = mid_entry.split('\t')
            id2mids[lineid].append(mid)
    return qids, id2mids


def pad_sentences(sentence, length, padding_word="<pad>"):
    num_padding = length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    return new_sentence


def get_pretrained_word_vector(file_path, dim):
    word_dict = {}
    embedding_matrix = np.zeros((dim[0] + 1, dim[1]))
    with open(file_path) as f:
        for idx, lines in enumerate(f):
            word_dict[lines.split()[0]] = idx
            embedding_matrix[idx] = np.array([float(item) for item in lines.split()[1:]])
        word_dict['<pad>'] = dim[0]  # <unk>
    return word_dict, embedding_matrix


def rela2idx(relation):
    rela_split = relation[3:].replace('_', '.').split('.')
    return rela_split


def load_index(filename):
    print("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index


def preprocess(index_reach, data_path, output_dir, ent_path, hits_ent=100):
    _, id2mids = get_mids(ent_path, hits_ent)
    qids, id2questions, id2goldrelas = get_questions(data_path)
    results_file = open(os.path.join(output_dir, "rela.top50el.test"), 'w')
    hit, max_cad_rela = 0, 0  # test top100 entity: 312
    for qid in qids:
        cand_relas = set()  # set
        question = id2questions[qid]
        gold_rela = id2goldrelas[qid]
        cand_mids = id2mids[qid]
        for mid in cand_mids:
            link_rel = index_reach[mid]
            cand_relas = cand_relas | set(link_rel)
        max_cad_rela = max(max_cad_rela, len(cand_relas))
        if gold_rela in cand_relas:
            hit += 1
        for cand_rela in set(cand_relas):
            results_file.write("{} %%%% {} %%%% {} %%%% {}\n".format(qid, question, gold_rela, cand_rela))
    print(max_cad_rela)
    print(hit, len(qids), float(hit / len(qids)))


def load_data(rela_voc_path, data_path, neg_size=50):
    qids, id2questions, id2goldrelas = get_questions(data_path)
    ques_list, rela_list = list(), list()
    max_ques_len, max_rela_len = 0, 0
    label = list()
    rela_voc = load_index(rela_voc_path)

    for qid in qids:
        temp_list = list()
        question = id2questions[qid]
        gold_rela = id2goldrelas[qid][3:]  # fb:
        gold_rela_split = clean_str(gold_rela).split(" ")
        temp_list.append(gold_rela_split)
        while len(temp_list) < (neg_size + 1):
            rand_idx = random.randint(0, len(rela_voc) - 1)
            rela_split = clean_str(rela_voc[rand_idx][3:]).split(" ")
            if rela_split not in temp_list:
                temp_list.append(rela_split)
        random.shuffle(temp_list)
        rela_list.append(temp_list)

        ques_split = clean_str(question).split(" ")
        max_ques_len = max(max_ques_len, len(ques_split))
        ques_list.append(ques_split)

        for rela in temp_list:
            if rela == gold_rela_split:
                idx = temp_list.index(rela)
                label.append(idx)
    print("max_ques_len:", max_ques_len)
    # print("max_rela_len:", max_rela_len)  # 17
    print(len(qids), len(label))
    print(ques_list[1], rela_list[1])
    print(label[:5])
    return ques_list, rela_list, label, max_ques_len, max_rela_len


if __name__ == "__main__":
    parser = ArgumentParser(description='Perform evidence integration')
    parser.add_argument('--index_reachpath', type=str, default="../../indexes/reachability_2M.pkl",
                        help='path to the pickle for the reachability index')
    parser.add_argument('--index_relation', type=str, default="../../indexes/relation_sub_2M.pkl",
                        help='path to the pickle for the relation index')
    parser.add_argument('--data_path', type=str, default="../../data/processed_simplequestions_dataset/test.txt")
    # parser.add_argument('--ent_path', type=str, default="../../entity_linking/results/nn/test-h100.txt",
    #                     help='path to the entity linking results')
    parser.add_argument('--hits_ent', type=int, default=50,
                        help='the hits here has to be <= the hits in entity linking')
    parser.add_argument('--output_dir', type=str, default="./results")
    args = parser.parse_args()
    print(args)

    relation_voc = load_index(args.index_relation)
    # index_reach = load_index(args.index_reachpath)
    load_data(relation_voc, args.data_path)
    # print "process valid data:"
