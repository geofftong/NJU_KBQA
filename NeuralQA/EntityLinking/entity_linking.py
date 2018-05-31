import os
from argparse import ArgumentParser
from collections import defaultdict

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from util import www2fb, get_ngram, get_stat_inverted_index

stopword = set(stopwords.words('english'))
inverted_index = defaultdict(list)


def entity_linking(predicted_file, gold_file, hits, output):
    predicted = open(predicted_file)
    gold = open(gold_file)
    fout = open(output, 'w')
    total, top1, top3, top5, top10, top20, top50, top100 = 0, 0, 0, 0, 0, 0, 0, 0
    for idx, (line, gold_id) in tqdm(enumerate(zip(predicted.readlines(), gold.readlines()))):
        total += 1
        line = line.strip().split(" %%%% ")
        gold_id = gold_id.strip().split('\t')[1]
        cand_entity, cand_score = [], []
        line_id = line[0]
        if len(line) == 2:
            tokens = get_ngram(line[1])
        else:
            tokens = []

        if len(tokens) > 0:
            maxlen = len(tokens[0].split())  # 1, 2, 3
        # print(maxlen)
        for item in tokens:  # todo
            if len(item.split()) < maxlen and len(cand_entity) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(cand_entity) > 0:
                break
            if item in stopword:
                continue
            cand_entity.extend(inverted_index[item])
            print(item)
            print(inverted_index[item])  # all name/alias contain 'item'(string)
            # if len(cand_entity) > 0:
            #     break
        print(cand_entity)
        for mid_text_type in sorted(set(cand_entity)):
            score = fuzz.ratio(mid_text_type[1], line[1]) / 100.0
            cand_score.append((mid_text_type, score))

        cand_score.sort(key=lambda t: t[1], reverse=True)
        cand_mids = cand_score[:hits]
        fout.write("{}".format(line_id))
        for mid_text_type, score in cand_mids:
            fout.write(" %%%% {}\t{}\t{}\t{}".format(mid_text_type[0], mid_text_type[1], mid_text_type[2], score))
        fout.write('\n')
        gold_id = www2fb(gold_id)
        mids_list = [x[0][0] for x in cand_mids]
        if gold_id in mids_list[:1]:
            top1 += 1
        if gold_id in mids_list[:3]:
            top3 += 1
        if gold_id in mids_list[:5]:
            top5 += 1
        if gold_id in mids_list[:10]:
            top10 += 1
        if gold_id in mids_list[:20]:
            top20 += 1
        if gold_id in mids_list[:50]:
            top50 += 1
        if gold_id in mids_list[:100]:
            top100 += 1

    print("total: {}".format(total))
    print("Top1 Entity Linking Accuracy: {}".format(top1 / total))
    print("Top3 Entity Linking Accuracy: {}".format(top3 / total))
    print("Top5 Entity Linking Accuracy: {}".format(top5 / total))
    print("Top10 Entity Linking Accuracy: {}".format(top10 / total))
    print("Top20 Entity Linking Accuracy: {}".format(top20 / total))
    print("Top50 Entity Linking Accuracy: {}".format(top50 / total))
    print("Top100 Entity Linking Accuracy: {}".format(top100 / total))


if __name__ == "__main__":
    # print(get_ngram("Which team have LeBron played basketball ?"))
    parser = ArgumentParser(description='Perform entity linking')
    parser.add_argument('--model_type', type=str, required=True, help="options are [crf|lstm|gru]")
    parser.add_argument('--index_ent', type=str, default="../indexes/entity_2M.pkl",
                        help='path to the pickle for the inverted entity index')
    parser.add_argument('--data_dir', type=str, default="../data/processed_simplequestions_dataset")
    parser.add_argument('--query_dir', type=str, default="../entity_detection/nn/query_text")
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default="./results")
    args = parser.parse_args()
    print(args)

    model_type = args.model_type.lower()
    # assert(model_type == "crf" or model_type == "lstm" or model_type == "gru")
    output_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)

    get_stat_inverted_index(args.index_ent)
    print("valid result:")
    entity_linking(
        os.path.join(args.query_dir, "query.valid"),
        os.path.join(args.data_dir, "valid.txt"),
        args.hits,
        os.path.join(output_dir, "valid-h{}.txt".format(args.hits)))

    print("test result:")
    entity_linking(
        os.path.join(args.query_dir, "query.test"),
        os.path.join(args.data_dir, "test.txt"),
        args.hits,
        os.path.join(output_dir, "test-h{}.txt".format(args.hits)))
