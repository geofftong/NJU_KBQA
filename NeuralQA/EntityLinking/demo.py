import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm

from util import www2fb

inverted_index = defaultdict(list)
stopword = set(stopwords.words('english'))


def get_ngram(text):
    # ngram = set()
    ngram = []
    tokens = text.split()
    for i in range(len(tokens) + 1):
        for j in range(i):
            if i - j <= 3:  # todo
                # ngram.add(" ".join(tokens[j:i]))
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    # ngram = list(ngram)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
    return ngram


def get_stat_inverted_index(filename):
    """
    Get the number of entry and max length of the entry (How many mid in an entry)
    """
    with open(filename, "rb") as handler:
        global inverted_index
        inverted_index = pickle.load(handler)
        inverted_index = defaultdict(str, inverted_index)
    # print("Total type of text: {}".format(len(inverted_index)))
    max_len = 0
    _entry = ""
    for entry, value in inverted_index.items():
        if len(value) > max_len:
            max_len = len(value)
            _entry = entry
    # print("Max Length of entry is {}, text is {}".format(max_len, _entry))


def entity_linking(pred_mention, top_num):
    C = []
    C_scored = []
    tokens = get_ngram(pred_mention)

    if len(tokens) > 0:
        maxlen = len(tokens[0].split())
    for item in tokens:
        if len(item.split()) < maxlen and len(C) == 0:
            maxlen = len(item.split())
        if len(item.split()) < maxlen and len(C) > 0:
            break
        if item in stopword:
            continue
        C.extend(inverted_index[item])
        # if len(C) > 0:
        #     break
    for mid_text_type in sorted(set(C)):
        score = fuzz.ratio(mid_text_type[1], pred_mention) / 100.0
        # C_counts format : ((mid, text, type), score_based_on_fuzz)
        C_scored.append((mid_text_type, score))

    C_scored.sort(key=lambda t: t[1], reverse=True)
    cand_mids = C_scored[:top_num]
    for mid_text_type, score in cand_mids:
        print("{}\t{}\t{}\t{}\t{}".format(pred_mention, mid_text_type[0], mid_text_type[1], mid_text_type[2], score))

if __name__ == "__main__":
    parser = ArgumentParser(description='Perform entity linking')
    parser.add_argument('--model_type', type=str, required=True, help="options are [crf|lstm|gru]")
    parser.add_argument('--index_ent', type=str, default="indexes/entity_2M.pkl",
                        help='path to the pickle for the inverted entity index')
    parser.add_argument('--data_dir', type=str, default="data/processed_simplequestions_dataset")
    parser.add_argument('--query_dir', type=str, default="entity_detection/crf/query_text")
    parser.add_argument('--hits', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="./results")

    # added for demo
    # parser.add_argument('--input_mention', type=str, default='yao ming')
    parser.add_argument('--input_path', type=str, default='')
    args = parser.parse_args()
    # print(args)

    input_mentions = open(args.input_path).read().strip()
    # print("input_manetion:", input_mention)

    get_stat_inverted_index(args.index_ent)
    for mention in input_mentions.split():
        entity_linking(mention, args.hits)
