from collections import defaultdict


def preprocess(mention, query):  # "you", "how are you"
    new_query = query.replace(mention, "<s>")
    print(new_query)
    return new_query


def replace_mention(filepath, output_path):
    with open(filepath) as f:
        with open(output_path, "w") as fw:
            not_match = 0
            for idx, line in enumerate(f):
                tokens = line.split("	")
                qid = tokens[0]
                sub = tokens[1]
                mention = tokens[2]
                relation = tokens[3]  # qid =
                obj = tokens[4]
                query = tokens[5]
                tag = tokens[6]
                if mention not in query:
                    not_match += 1
                new_query = query.replace(mention, "<s>")
                fw.write(
                    qid + "\t" + sub + "\t" + mention + "\t" + relation + "\t" + obj + "\t" + new_query + "\t" + tag)
    print("num", not_match)


def get_mid2wiki(filepath):
    print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filepath)
    idx = 0
    for line in fin.readlines():
        items = line.strip().split('\t')
        if len(items) != 3:
            continue
        else:
            idx += 1
            url = items[2]
            print(idx, url[1:-3])
            # sub = rdf2fb(clean_uri(items[0]))
            # mid2wiki[sub] = True
    return mid2wiki


if __name__ == "__main__":
    # filepath = "../../data/processed_simplequestions_dataset/train.txt"
    # replace_mention(filepath)
    # get_mid2wiki("../../data/fb2w.nt")
    # result = preprocess("second battle of fort fisher",
    #                     "which military was involved in the second battle of fort fisher in China")
    replace_mention("../../data/processed_simplequestions_dataset/test.txt", "test_relation")
