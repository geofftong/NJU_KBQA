import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict

import math

from util import clean_uri, www2fb, rdf2fb


# Load up reachability graph

def load_index(filename):
    # print("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index


# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    # print("Entity Source : {}".format(filename))
    id2mids = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0]
        cand_mids = items[1:][:hits]
        for mid_entry in cand_mids:
            # TODO:WHY MID_TYPE ONLY ONE! TYPE IS FROM CFO NAME EITHER TYPE.TOPIC.NAME OR COMMON.TOPIC.ALIAS
            mid, mid_name, mid_type, score = mid_entry.split('\t')
            id2mids[lineid].append((mid, mid_name, mid_type, float(score)))
    return id2mids


def get_rels(filename, hits):
    # print("Relation Source : {}".format(filename))
    id2rels = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0].strip()
        rel = www2fb(items[1].strip())
        label = items[2].strip()
        score = items[3].strip()
        if len(id2rels[lineid]) < hits:
            id2rels[lineid].append((rel, label, float(score)))
    return id2rels


def get_questions(filename):
    # print("getting questions ...")
    id2questions = {}
    id2goldmids = {}
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        lineid = items[0].strip()
        mid = items[1].strip()
        question = items[5].strip()
        rel = items[3].strip()
        id2questions[lineid] = (question, rel)
        id2goldmids[lineid] = mid
    return id2questions, id2goldmids


def get_mid2wiki(filename):
    # print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki


def evidence_integration(index_reach, index_degrees, mid2wiki, is_heuristics, input_entity, input_relation):
    id2answers = list()
    mids = input_entity.split("\n")
    rels = input_relation.split("\n")
    # print(rels)

    if is_heuristics:
        for item in mids:
            _, mid, mid_name, mid_type, mid_score = item.strip().split("\t")
            for item2 in rels:
                rel, rel_log_score = item2.strip().split("\t")
                # if this (mid, rel) exists in FB
                if rel in index_reach[mid]:
                    rel_score = math.exp(float(rel_log_score))
                    comb_score = (float(mid_score) ** 0.6) * (rel_score ** 0.1)
                    id2answers.append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score,
                                       int(mid2wiki[mid]), int(index_degrees[mid][0])))
                    # I cannot use retrieved here because I use contain different name_type
                    # if mid ==truth_mid and rel == truth_rel:
                    #     retrieved += 1
        id2answers.sort(key=lambda t: (t[6], t[3], t[7], t[8]), reverse=True)
    else:
        id2answers = [(mids[0][0], rels[0][0])]

    # write to file
    # TODO:CHANGED FOR SWITCH IS_HEURISTICS
    if is_heuristics:
        for answer in id2answers:
            mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, _, _ = answer
            print("{}\t{}\t{}\t{}\t{}".format(mid, rel, mid_name, mid_score, rel_score, comb_score))
    else:
        for answer in id2answers:
            mid, rel = answer
            print("{}\t{}".format(mid, rel))
    return id2answers


if __name__ == "__main__":
    parser = ArgumentParser(description='Perform evidence integration')
    parser.add_argument('--ent_type', type=str, required=True, help="options are [crf|lstm|gru]")
    parser.add_argument('--rel_type', type=str, required=True, help="options are [lr|cnn|lstm|gru]")
    parser.add_argument('--index_reachpath', type=str, default="indexes/reachability_2M.pkl",
                        help='path to the pickle for the reachability index')
    parser.add_argument('--index_degreespath', type=str, default="indexes/degrees_2M.pkl",
                        help='path to the pickle for the index with the degree counts')
    parser.add_argument('--data_path', type=str, default="data/processed_simplequestions_dataset/test.txt")
    parser.add_argument('--ent_path', type=str, default="entity_linking/results/lstm/test-h100.txt",
                        help='path to the entity linking results')
    parser.add_argument('--rel_path', type=str, default="relation_prediction/nn/results/cnn/test.txt",
                        help='path to the relation prediction results')
    parser.add_argument('--wiki_path', type=str, default="data/fb2w.nt")
    parser.add_argument('--hits_ent', type=int, default=50,
                        help='the hits here has to be <= the hits in entity linking')
    parser.add_argument('--hits_rel', type=int, default=5,
                        help='the hits here has to be <= the hits in relation prediction retrieval')
    parser.add_argument('--no_heuristics', action='store_false', help='do not use heuristics', dest='heuristics')
    parser.add_argument('--output_dir', type=str, default="./results")

    # added for demo
    parser.add_argument('--input_ent_path', type=str, default='el_result.txt')
    parser.add_argument('--input_rel_path', type=str, default='rp_result.txt')
    args = parser.parse_args()
    # print(args)

    ent_type = args.ent_type.lower()
    rel_type = args.rel_type.lower()
    output_dir = os.path.join(args.output_dir, "{}-{}".format(ent_type, rel_type))
    os.makedirs(output_dir, exist_ok=True)

    index_reach = load_index(args.index_reachpath)
    # print(index_reach)
    index_degrees = load_index(args.index_degreespath)
    mid2wiki = get_mid2wiki(args.wiki_path)

    candidate_entity = open(args.input_ent_path).read().strip()
    candidate_relation = open(args.input_rel_path).read().strip()
    test_answers = evidence_integration(index_reach, index_degrees, mid2wiki, args.heuristics, candidate_entity,
                                        candidate_relation)
