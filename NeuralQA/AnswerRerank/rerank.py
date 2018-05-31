import os
from argparse import ArgumentParser
from collections import defaultdict

import math
from util import get_mid2wiki, get_rels, get_mids, get_questions, www2fb, load_index


def answer_rerank(data_path, ent_path, rel_path, output_dir, index_reach, index_degrees, mid2wiki, is_heuristics,
                         ent_hits, rel_hits):
    id2questions, id2goldmids = get_questions(data_path)
    id2mids = get_mids(ent_path, ent_hits)
    id2rels = get_rels(rel_path, rel_hits)
    file_base_name = os.path.basename(data_path)
    fout = open(os.path.join(output_dir, file_base_name), 'w')

    id2answers = defaultdict(list)
    found, notfound_both, notfound_mid, notfound_rel = 0, 0, 0, 0
    retrieved, retrieved_top1, retrieved_top2, retrieved_top3 = 0, 0, 0, 0
    lineids_found1 = []
    lineids_found2 = []
    lineids_found3 = []

    # for every lineid
    for line_id in id2goldmids:
        if line_id not in id2mids and line_id not in id2rels:
            notfound_both += 1
            continue
        elif line_id not in id2mids:
            notfound_mid += 1
            continue
        elif line_id not in id2rels:
            notfound_rel += 1
            continue
        found += 1
        question, truth_rel = id2questions[line_id]
        truth_rel = www2fb(truth_rel)
        truth_mid = id2goldmids[line_id]
        mids = id2mids[line_id]
        rels = id2rels[line_id]

        if is_heuristics:
            for (mid, mid_name, mid_type, mid_score) in mids:
                for (rel, rel_label, rel_log_score) in rels:
                    # if this (mid, rel) exists in FB
                    if rel in index_reach[mid]:
                        rel_score = math.exp(float(rel_log_score))
                        comb_score = (float(mid_score) ** 0.6) * (rel_score ** 0.1)
                        id2answers[line_id].append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score,
                                                    int(mid2wiki[mid]), int(index_degrees[mid][0])))
                        # I cannot use retrieved here because I use contain different name_type
                        # if mid ==truth_mid and rel == truth_rel:
                        #     retrieved += 1
            id2answers[line_id].sort(key=lambda t: (t[6], t[3], t[7], t[8]), reverse=True)
        else:
            id2answers[line_id] = [(mids[0][0], rels[0][0])]

        # write to file
        fout.write("{}".format(line_id))
        if is_heuristics:
            for answer in id2answers[line_id]:
                mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, _, _ = answer
                fout.write(" %%%% {}\t{}\t{}\t{}\t{}".format(mid, rel, mid_name, mid_score, rel_score, comb_score))
        else:
            for answer in id2answers[line_id]:
                mid, rel = answer
                fout.write(" %%%% {}\t{}".format(mid, rel))
        fout.write('\n')

        if is_heuristics:
            if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][1] == truth_rel:  # id2answers[line_id][0][0] == truth_mid and
                retrieved_top1 += 1
                retrieved_top2 += 1
                retrieved_top3 += 1
                lineids_found1.append(line_id)
            elif len(id2answers[line_id]) >= 2 and id2answers[line_id][1][0] == truth_mid \
                    and id2answers[line_id][1][1] == truth_rel:
                retrieved_top2 += 1
                retrieved_top3 += 1
                lineids_found2.append(line_id)
            elif len(id2answers[line_id]) >= 3 and id2answers[line_id][2][0] == truth_mid \
                    and id2answers[line_id][2][1] == truth_rel:
                retrieved_top3 += 1
                lineids_found3.append(line_id)
        else:
            if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][0] == truth_mid \
                    and id2answers[line_id][0][1] == truth_rel:
                retrieved_top1 += 1
                retrieved_top2 += 1
                retrieved_top3 += 1
                lineids_found1.append(line_id)
    print()
    print("found:              {}".format(found / len(id2goldmids) * 100.0))
    print("retrieved at top 1: {}".format(retrieved_top1 / len(id2goldmids) * 100.0))
    print("retrieved at top 2: {}".format(retrieved_top2 / len(id2goldmids) * 100.0))
    print("retrieved at top 3: {}".format(retrieved_top3 / len(id2goldmids) * 100.0))
    # print("retrieved at inf:   {}".format(retrieved / len(id2goldmids) * 100.0))
    fout.close()
    return id2answers


if __name__ == "__main__":
    parser = ArgumentParser(description='Perform evidence integration')
    parser.add_argument('--ent_type', type=str, required=True, help="options are [crf|lstm|gru]")
    parser.add_argument('--rel_type', type=str, required=True, help="options are [lr|cnn|lstm|gru]")
    parser.add_argument('--index_reachpath', type=str, default="../indexes/reachability_2M.pkl",
                        help='path to the pickle for the reachability index')
    parser.add_argument('--index_degreespath', type=str, default="../indexes/degrees_2M.pkl",
                        help='path to the pickle for the index with the degree counts')
    parser.add_argument('--data_path', type=str, default="../data/processed_simplequestions_dataset/test.txt")
    parser.add_argument('--ent_path', type=str, default="../entity_linking/results/crf/test-h100.txt",
                        help='path to the entity linking results')
    parser.add_argument('--rel_path', type=str, default="../relation_prediction/nn/results/cnn/test.txt",
                        help='path to the relation prediction results')
    parser.add_argument('--wiki_path', type=str, default="../data/fb2w.nt")
    parser.add_argument('--hits_ent', type=int, default=50,
                        help='the hits here has to be <= the hits in entity linking')
    parser.add_argument('--hits_rel', type=int, default=5,
                        help='the hits here has to be <= the hits in relation prediction retrieval')
    parser.add_argument('--no_heuristics', action='store_false', help='do not use heuristics', dest='heuristics')
    parser.add_argument('--output_dir', type=str, default="./results")
    args = parser.parse_args()
    print(args)

    ent_type = args.ent_type.lower()
    rel_type = args.rel_type.lower()
    # assert (ent_type == "crf" or ent_type == "lstm" or ent_type == "gru")
    # assert (rel_type == "lr" or rel_type == "cnn" or rel_type == "lstm" or rel_type == "gru")
    output_dir = os.path.join(args.output_dir, "{}-{}".format(ent_type, rel_type))
    os.makedirs(output_dir, exist_ok=True)

    index_reach = load_index(args.index_reachpath)
    index_degrees = load_index(args.index_degreespath)
    mid2wiki = get_mid2wiki(args.wiki_path)

    test_answers = answer_rerank(args.data_path, args.ent_path, args.rel_path, output_dir, index_reach,
                                        index_degrees, mid2wiki, args.heuristics, args.hits_ent, args.hits_rel)
