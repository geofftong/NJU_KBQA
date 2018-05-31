import unicodedata
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


# Load up reachability graph

def load_index(filename):
    print("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index


# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    print("Entity Source : {}".format(filename))
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
    print("Relation Source : {}".format(filename))
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
    print("getting questions ...")
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
    print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki


def processed_text(text):
    text = text.replace('\\\\', '')
    # stripped = strip_accents(text.lower())
    stripped = text.lower()
    toks = tokenizer.tokenize(stripped)
    return " ".join(toks)


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    if in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    # Manual Correction
    if in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    if in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    if in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    if in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    if in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    if in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    if in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    if in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    if in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    if in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    if in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    if in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    if in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return in_str


def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri


def rdf2fb(in_str):
    if in_str.startswith('http://rdf.freebase.com/ns/'):
        return 'fb:%s' % (in_str.split('http://rdf.freebase.com/ns/')[-1])
