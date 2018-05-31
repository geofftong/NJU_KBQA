import unicodedata
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


def readFile(filename):
    context = open(filename).readlines()
    return [c.strip() for c in context]


def writeFile(context, filename, append=False):
    if not append:
        with open(filename, 'w+') as fout:
            for co in context:
                fout.write(co + "\n")
    else:
        with open(filename, 'a+') as fout:
            for co in context:
                fout.write(co + "\n")


def list2str(l, split=" "):
    a = ""
    for li in l:
        a += (str(li) + split)
    a = a[:-len(split)]
    return a


def get_ngram(text):
    # ngram = set()
    ngram = []
    tokens = text.split()
    for i in range(len(tokens) + 1):
        for j in range(i):
            if i - j <= 3:  # 3 ?
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
    print("Total type of text: {}".format(len(inverted_index)))
    max_len = 0
    _entry = ""
    for entry, value in inverted_index.items():
        if len(value) > max_len:
            max_len = len(value)
            _entry = entry
    print("Max Length of entry is {}, text is {}".format(max_len, _entry))


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
