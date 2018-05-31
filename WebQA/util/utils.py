import json
import pickle
from collections import defaultdict

import unicodedata
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

stopwords = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()


def get_mid2wiki(filename):
    # print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    mid2url = defaultdict()
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        if len(items) != 3:
            continue
        else:
            sub = rdf2fb(clean_uri(items[0]))
            mid2wiki[sub] = True
            url = items[2][1:-3]
            mid2url[sub] = url
    return mid2wiki, mid2url


def convert_json(filepath):
    result = list()
    with open(filepath) as f:
        for line in f:
            test_data = dict()
            tokens = line.split("\t")
            test_data["id"] = tokens[0]
            test_data["relation"] = tokens[3]
            test_data["question"] = tokens[5]
            result.append(test_data)
    return result


def get_mid2name_alias(name_path, alias_path):
    print("loading data from: {}".format(name_path))
    with open(name_path, 'rb') as f:
        names = pickle.load(f)
    with open(alias_path, 'rb') as f:
        alias = pickle.load(f)
    return names, alias


def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens


def get_index(index_path):
    print("loading data from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


def get_ngram(text):
    # ngram = set()
    ngram = list()
    tokens = text.split()
    for i in range(len(tokens) + 1):
        for j in range(i):
            if i - j <= 3:  # todo: 3?
                # ngram.add(" ".join(tokens[j:i]))
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    # ngram = list(ngram)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
    return ngram


def pick_name(question, names_list):
    max_score = None
    predict_name = None
    for name in names_list:
        score = fuzz.ratio(name, question)
        if score > max_score:
            max_score = score
            predict_name = name
    return predict_name


def www2fb(in_str):
    out_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return out_str


def rdf2fb(in_str):
    out_str = 'fb:%s' % (in_str.split('http://rdf.freebase.com/ns/')[-1])
    return out_str


class ins(object):
    def __init__(self, question):
        self.question = question


def get_span(label):
    start, end = 0, 0
    flag = False
    span = []
    for k, l in enumerate(label):
        if l == 'I' and not flag:
            start = k
            flag = True
        if l != 'I' and flag:
            flag = False
            en = k
            span.append((start, en))
            start, end = 0, 0
    if start != 0 and end == 0:
        end = len(label) + 1  # bug fixed: geoff
        span.append((start, end))
    return span


def get_names(fb_names, cand_mids):
    names = list()
    for mid in cand_mids:
        if mid in fb_names:
            names.append(fb_names[mid][0])  # todo
    return names


def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri


if __name__ == '__main__':
    result = convert_json("../data/processed_dataset/test.txt")
    # Writing JSON data
    with open('data.json', 'w') as f:
        json.dump(result, f)

        # # Reading data back
        # with open('data.json', 'r') as f:
        #     data = json.load(f)
