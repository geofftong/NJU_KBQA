import random
import re

import numpy as np


def read_relation():
    relation_dict, type_dict, entity2relation = {}, {}, {}
    with open("../data/FB5M/FB5M.relation.vocabulary") as f:
        for line in f:
            tokens = line.split("\t")
            relation_dict[tokens[0]] = tokens[1].strip()
    with open("../data/FB5M/FB5M.subject2relation") as f:
        for line in f:
            relation_list = []
            tokens = line.split("\t")
            relation = tokens[1].split()
            for i in range(len(relation)):
                relation_list.append(relation_dict[relation[i].strip()])  # relation_dict[relation[i].strip()]
            entity2relation[tokens[0]] = relation_list
    # print len(entity2relation), entity2relation["m.07cd9t"]
    return relation_dict, entity2relation


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


def save_data(input_path, output_path, relation_score_list, relation_list):  # TODO: IN = ONT
    print(np.array(relation_score_list).shape, np.array(relation_list).shape)
    print(relation_score_list[0])
    print(relation_list[0])  # 20545, 300, 36
    relation_score_dict_list = []
    for qid in range(len(relation_score_list)):  # 0-21687
        relation_score_dict = {}
        for idx in range(len(relation_score_list[qid])):
            relation_score_dict['.'.join(relation_list[qid][idx])] = relation_score_list[qid][idx]
        relation_score_dict_list.append(relation_score_dict)

    with open(input_path) as f_in:
        with open(output_path, "w") as f_out:
            for line in f_in:
                tokens = line.split("\t")
                qid = tokens[0]
                linked_relation = tokens[6].strip().replace('/', '.').replace('_', '.')
                # print qid, len(relation_score_dict_list)
                # if int(qid) >= len(relation_score_dict_list):
                #     break
                if linked_relation in relation_score_dict_list[int(qid)]:
                    score = relation_score_dict_list[int(qid)][linked_relation]
                    f_out.write(line.strip() + "\t" + str(score) + "\n")


def load_data(file_path, train_flag=True, neg_sample=300):  # train(contains pos):300 valid:302 test:300(294)
    max_len, real_label = 0, []
    rela_dict, _ = read_relation()
    sample_ques_dict, sample_rela_dict = {}, {}  # TODO: remove
    sample_ques_list, sample_rela_list = [], []
    with open(file_path) as f:
        for line in f:
            tokens = line.split("\t")
            qid, gold_question, gold_relation = tokens[0], tokens[1], tokens[5]
            linked_relation = tokens[6].strip()
            if qid not in sample_ques_dict:
                sample_ques_dict[qid] = [gold_question]
            if qid not in sample_rela_dict:  # add gold relation
                if train_flag:
                    sample_rela_dict[qid] = [gold_relation]
                else:  # Trick: random real label index (1142 miss gold relation) for Test set
                    sample_rela_dict[qid] = []
                    real_label.append(random.randint(0, neg_sample - 1))
            if linked_relation != 'null' and (linked_relation not in sample_rela_dict[qid]):  # add linked relation
                if len(sample_rela_dict[qid]) < neg_sample:
                    sample_rela_dict[qid].append(linked_relation)
                    if not train_flag and linked_relation == gold_relation:
                        real_label[int(qid)] = sample_rela_dict[qid].index(linked_relation)

    for qid in sample_rela_dict:  # train need negative sampling, test need padding zeros
        while len(sample_rela_dict[qid]) < neg_sample:
            if train_flag:
                rand_idx = random.randint(0, len(rela_dict) - 1)
                if rela_dict[str(rand_idx)] not in sample_rela_dict[qid]:
                    sample_rela_dict[qid].append(rela_dict[str(rand_idx)])
            else:
                sample_rela_dict[qid].append("unk")
    for qid in range(len(sample_rela_dict)):
        ques_split = clean_str(''.join(sample_ques_dict[str(qid)])).split(" ")
        sample_ques_list.append(ques_split)
        rela_list = []
        for rela in sample_rela_dict[str(qid)]:
            temp_split = clean_str(rela).split(" ")
            max_len = max(max_len, len(temp_split), len(ques_split))
            rela_list.append(temp_split)
        sample_rela_list.append(rela_list)
        if train_flag:
            real_label.append(random.randint(0, len(sample_rela_dict[str(qid)]) - 1))
            sample_rela_list[qid][0], sample_rela_list[qid][real_label[qid]] = sample_rela_list[qid][real_label[qid]], \
                                                                               sample_rela_list[qid][0]
    # print len(sample_rela_dict), len(sample_rela_list), max_len
    return sample_ques_list, sample_rela_list, real_label, max_len


def preprocess(gold_data_path, entity_score_path, output_data_path, train_flag=True, neg_sample=300, entity_num=20):
    _, entity2relation = read_relation()
    with open(gold_data_path) as f:
        gold_list = []
        for lines in f:
            data = lines.split('\t')
            gold_entity = data[0][17:].replace('/', '.')
            gold_relation = data[1][17:].replace('/', '.')
            gold_answer = data[2][17:].replace('/', '.')
            gold_question = data[3].strip()
            gold_list.append([gold_entity, gold_relation, gold_answer, gold_question])
    with open(entity_score_path) as f:
        candidate_entity_score_list = []
        for lines in f:
            data = lines.strip().split('\t')
            length = len(data) if len(data) < entity_num + 1 else entity_num + 1  # top 20 entities
            if length == 1:  # test:741
                candidate_entity_score_list.append("")
            elif length > 1:
                temp_dict = {}
                for i in range(1, length):
                    tokens = data[i].split()
                    entity_id = tokens[0]
                    entity_score = tokens[1]
                    temp_dict[entity_id] = entity_score
                candidate_entity_score_list.append(temp_dict)
    with open(output_data_path, "w") as f:
        max_relation, unsolved_count = 0, 0
        for idx in range(len(gold_list)):
            count = 0
            if not train_flag and gold_list[idx][0] not in candidate_entity_score_list[idx]:
                unsolved_count += 1
                # continue
            if candidate_entity_score_list[idx] == "":  # qid 741 in test, no el result
                f.write(str(idx) + "\t" + gold_list[idx][3] + "\t" + gold_list[idx][
                    0] + "\t" + "null" + "\t" + "null" + "\t" + gold_list[idx][1] + "\t" + "null" + "\t" + "\n")
                continue
            for entity in candidate_entity_score_list[idx]:
                entity_score = candidate_entity_score_list[idx][entity]
                if train_flag and count >= neg_sample:
                    break
                if entity not in entity2relation:  # some entity_id not in entity2relation
                    f.write(
                        str(idx) + "\t" + gold_list[idx][3] + "\t" + gold_list[idx][
                            0] + "\t" + entity + "\t" + entity_score
                        + "\t" + gold_list[idx][1] + "\t" + "null" + "\t" + "\n")  # relation = 'null'
                    count += 1
                else:
                    for relation in entity2relation[entity]:
                        if train_flag and count + 1 >= neg_sample:  # contains one postive sample
                            break
                        f.write(
                            str(idx) + "\t" + gold_list[idx][3] + "\t" + gold_list[idx][
                                0] + "\t" + entity + "\t" + entity_score
                            + "\t" + gold_list[idx][1] + "\t" + relation + "\t" + "\n")
                        count += 1
            max_relation = count if count > max_relation else max_relation
        print("max count of candidate linked relation: %d" % max_relation)
        print("count of gold entity miss: %d" % unsolved_count)  # 2516


def pad_sentences(sentence, length, padding_word="unk"):
    num_padding = length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    return new_sentence


def get_pretrained_word_vector(file_path, dim):
    word_dict = {}
    embedding_matrix = np.zeros((dim[0] + 1, dim[1]))
    with open(file_path) as f:
        f.next()
        for idx, lines in enumerate(f):
            word_dict[lines.split()[0]] = idx
            embedding_matrix[idx] = np.array(map(float, lines.split()[1:]))
        word_dict['unk'] = dim[0]
        # print word_dict['unk'], embedding_matrix[dim[0]]
    return word_dict, embedding_matrix


if __name__ == "__main__":
    gold_data_path_list = ['../../data/SimpleQuestions/annotated_fb_data_train.txt',
                           '../../data/SimpleQuestions/annotated_fb_data_valid.txt',
                           '..././data/SimpleQuestions/annotated_fb_data_test.txt']
    entity_score_path_list = ['../../data/SimpleQuestions/train.fuzzy_p2_linker.simple_linker.original.union',
                              '../../data/SimpleQuestions/valid.fuzzy_p2_linker.simple_linker.original.union',
                              '../../data/SimpleQuestions/test.fuzzy_p2_linker.simple_linker.original.union']
    output_data_path_list = ["../../data/SimpleQuestions/output/simple.train.top20el.top100relation",  # top300
                             "../../data/SimpleQuestions/output/simple.valid.top20el.top300relation",
                             "../../data/SimpleQuestions/output/simple.test.top20el.top300relation"]
    print("process train data:")
    preprocess(gold_data_path_list[0], entity_score_path_list[0], output_data_path_list[0], True, 100)
    # print "process valid data:"
    # preprocess(gold_data_path_list[1], entity_score_path_list[1], output_data_path_list[1], False)
    # print "process test data:"
    # preprocess(gold_data_path_list[2], entity_score_path_list[2], output_data_path_list[2], False)

    # start = time.time()
    # question_test_list, relation_test_list, label_test, _ = load_data(
    #     "../data/SimpleQuestions/output/simple.test.top20el.relation", False)  # 294
    # cnt = 0
    # for label in label_test:
    #     if label == -1:
    #         cnt += 1
    # end = time.time()
    # print 'running time: %s' % str(end - start)
    # print 'test: %d, %f' % (cnt, cnt * 1.0 / len(label_test))
