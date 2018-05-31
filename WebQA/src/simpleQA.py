import math
import numpy as np
import torch
from args import get_kb_args, get_relation_args, get_entity_args
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from torchtext import data
from util.datasets import SimpleQADataset, SimpleQaRelationDataset
from util.utils import get_mid2wiki, get_mid2name_alias, get_ngram, ins, get_span, get_names, get_index

stopwords = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()


class SimpleQA:
    def __init__(self):
        kb_args = get_kb_args()
        self.index_ent = get_index(kb_args.idx_entity)
        self.index_names = get_index(kb_args.idx_name)
        self.index_reach = get_index(kb_args.idx_reachability)
        self.index_degrees = get_index(kb_args.idx_degree)
        self.fb_graph = get_index(kb_args.idx_freebase)
        self.mid2wiki, self.mid2url = get_mid2wiki(kb_args.wiki_path)

        # self.mid2name, self.mid2alias = get_mid2name_alisa(kb_args.name_path, kb_args.alias_path)

        # self.setup()

    def setup(self):
        entity_args = get_entity_args()
        relation_args = get_relation_args()

        torch.manual_seed(entity_args.seed)
        if not entity_args.cuda:
            entity_args.gpu = -1
        if torch.cuda.is_available() and entity_args.cuda:
            torch.cuda.set_device(entity_args.gpu)
            torch.cuda.manual_seed(entity_args.seed)

        # for entity detection
        questions_ent = data.Field(lower=True, sequential=True)
        labels = data.Field(sequential=True)
        train, dev, test = SimpleQADataset.splits(questions_ent, labels, path=entity_args.data_dir)
        questions_ent.build_vocab(train, dev, test)
        labels.build_vocab(train, dev, test)
        index2tag = np.array(labels.vocab.itos)
        # index2word = np.array(questions_ent.vocab.itos)
        print("entity vocab num: ", len(questions_ent.vocab.itos))

        # TODO: entity_model embedding
        # match_embedding = 0
        # if os.path.isfile(entity_args.vector_cache):
        #     stoi, vectors, dim = torch.load(entity_args.vector_cache)  # stoi?
        #     questions_ent.vocab.vectors = torch.Tensor(len(questions_ent.vocab), dim)
        #     for i, token in enumerate(questions_ent.vocab.itos):
        #         wv_index = stoi.get(token, None)
        #         if wv_index is not None:
        #             questions_ent.vocab.vectors[i] = vectors[wv_index]
        #             match_embedding += 1
        #         else:
        #             questions_ent.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
        # else:
        #     print("Error: Need word embedding pt file")
        #     exit(1)
        # print("Embedding match number {} out of {}".format(match_embedding, len(questions_ent.vocab)))

        questions_rel = data.Field(lower=True)  # , tokenize=tokenize_text
        relations = data.Field(sequential=False)
        train, dev, test = SimpleQaRelationDataset.splits(questions_rel, relations, path=relation_args.data_dir)
        questions_rel.build_vocab(train, dev, test)
        relations.build_vocab(train, dev)  # todo: bugfixed: test
        index2rel = np.array(relations.vocab.itos)
        print("relation vocab num: ", len(questions_rel.vocab.itos))

        # # TODO: relation_model embedding
        # if os.path.isfile(relation_args.vector_cache):
        #     stoi, vectors, dim = torch.load(relation_args.vector_cache)  # stoi?
        #     questions_rel.vocab.vectors = torch.Tensor(len(questions_rel.vocab), dim)
        #     for i, token in enumerate(questions_rel.vocab.itos):
        #         wv_index = stoi.get(token, None)
        #         if wv_index is not None:
        #             questions_rel.vocab.vectors[i] = vectors[wv_index]
        #         else:
        #             questions_rel.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
        # else:
        #     print("Error: Need word embedding pt file")
        #     exit(1)

        # load the model
        if entity_args.gpu == -1:
            model_ent = torch.load(entity_args.trained_model)
            model_rel = torch.load(relation_args.trained_model)
        else:
            model_ent = torch.load(entity_args.trained_model,
                                   map_location=lambda storage, location: storage.cuda(relation_args.gpu))
            model_rel = torch.load(relation_args.trained_model,
                                   map_location=lambda storage, location: storage.cuda(relation_args.gpu))
        print(model_ent)
        print(model_rel)
        # if relation_args.word_vectors:
        #     model_ent.embed.weight.data.copy_(questions_ent.vocab.vectors)
        #     model_rel.embed.weight.data.copy_(questions_rel.vocab.vectors)
        # print("**********************")
        self.questions_ent = questions_ent
        self.questions_rel = questions_rel
        self.model_rel = model_rel
        # print(model_rel)
        self.model_ent = model_ent
        # print(model_ent)
        self.index2rel = index2rel
        # print("index2rel:", len(index2rel))
        # print(index2rel)
        self.index2tag = index2tag
        # print("index2tag:", len(index2tag))
        # print(index2tag)
        self.args = relation_args

    def get_mentions(self, input_sent, questions, ent_model, index2tag, args):
        sent = tokenizer.tokenize(input_sent.lower())
        sent_raw = tokenizer.tokenize(input_sent)  # bugfixed: mention
        example = ins(questions.numericalize(questions.pad([sent]), device=args.gpu, train=False))
        ent_model.eval()
        scores = ent_model(example)
        index_tag = np.transpose(torch.max(scores, 1)[1].cpu().data.numpy())
        tag_array = index2tag[index_tag]
        spans = get_span(tag_array)  # return a list
        query_tokens = []
        for span in spans:
            query_tokens.append(" ".join(sent_raw[span[0]:span[1]]))
        if not query_tokens:
            # query_tokens = list(input_sent)
            query_tokens = input_sent.split()
        # print("candidate mention: \t{}".format(' ||| '.join(query_tokens)))
        query_tokens = sorted(query_tokens, key=lambda x: len(x.split()), reverse=True)
        return query_tokens  # return a mention list

    def get_relations(self, input_sent, questions, model, index2rel, args):
        cadidate_relations = list()
        sent = tokenizer.tokenize(input_sent.lower())
        example = ins(questions.numericalize(questions.pad([sent]), device=args.gpu, train=False))
        print("example:")
        model.eval()
        scores = model(example)
        # Get top k
        top_k_scores, top_k_indices = torch.topk(scores, k=self.args.hits, dim=1, sorted=True)  # shape: (batch_size, k)
        print(top_k_scores, top_k_indices)
        top_k_scores_array = top_k_scores.cpu().data.numpy()
        top_k_relatons_array = index2rel[top_k_indices.cpu().data.numpy()]

        print("\ncandidate relation:")
        for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
            for j, (rel, score) in enumerate(zip(relations_row, scores_row)):
                cadidate_relations.append((rel, score))
        if len(cadidate_relations) > 0:
            print(cadidate_relations[0])
        return cadidate_relations

    def get_answer(self, question):
        query_tokens = self.get_mentions(question, self.questions_ent, self.model_ent, self.index2tag, self.args)
        query_tokens = query_tokens[0]  # default: choose the longest ngram as mention
        abstract_ques = question.replace(query_tokens, "")  # todo
        print("abstract question:", abstract_ques)
        pred_relation = self.get_relations(question, self.questions_rel, self.model_rel, self.index2rel, self.args)

        # entity linking
        cand_entity = []  # candidate entities
        cand_score = []  # add score
        ngrams_tokens = get_ngram(query_tokens.lower())  # bugfixed: lower()
        print("ngram_tokens:")
        print(ngrams_tokens)
        if len(ngrams_tokens) > 0:
            maxlen = len(ngrams_tokens[0].split())
        for item in ngrams_tokens:  # todo: maxlen
            if len(item.split()) < maxlen and len(cand_entity) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(cand_entity) > 0:
                break
            if item in stopwords:
                continue
            if item in self.index_ent:  # bugfixed: unk
                cand_entity.extend(self.index_ent[item])

        for mid_text_type in sorted(set(cand_entity)):
            score = fuzz.ratio(mid_text_type[1], query_tokens.lower()) / 100.0  # todo: may have bug
            cand_score.append((mid_text_type, score))

        cand_score.sort(key=lambda t: t[1], reverse=True)
        print("\ncandidate entity:")
        if len(cand_score) > 0:
            print(cand_score[0])  # cand_score[:top_num]

        # rerank
        id2answers = list()
        for item in cand_score:
            (mid, mid_name, mid_type), mid_score = item
            for item2 in pred_relation:
                rel, rel_log_score = item2  # item2.strip().split("\t")
                if rel in self.index_reach[mid]:
                    rel_score = math.exp(float(rel_log_score))
                    comb_score = (float(mid_score) ** 0.6) * (rel_score ** 0.1)
                    id2answers.append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score,
                                       int(self.mid2wiki[mid]), int(self.index_degrees[mid][0])))
        id2answers.sort(key=lambda t: (t[6], t[3], t[7], t[8]), reverse=True)

        print("\nresult after reranking:")
        result = dict()
        result['answer'] = list()
        result['flag'] = True
        for item in id2answers:
            mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, _, _ = item
            key = (mid, rel)
            if key not in self.fb_graph:
                print("wrong key!")
                continue
            else:
                result_mid = self.fb_graph[key]
                # print(result_mid)
                names = get_names(self.index_names, result_mid)
                if len(names) == 0:
                    continue
                else:
                    answer = '&&'.join(names)
            if list(result_mid)[0] in self.mid2url:
                # print("result_mid:")
                # print(self.mid2url[list(result_mid)[0]])
                result['answer'].append(
                    ((answer, "%.5f" % comb_score, mid_name, "%.5f" % mid_score, rel, "%.5f" % rel_score, question,
                      query_tokens, self.mid2url[list(result_mid)[0]])))
            else:
                result['answer'].append(
                    ((answer, "%.5f" % comb_score, mid_name, "%.5f" % mid_score, rel, "%.5f" % rel_score, question,
                      query_tokens, "no wiki entity found")))
        if len(result['answer']) > 0:
            print(result['answer'][0])
        if len(result['answer']) == 0:
            result['flag'] = False
            result['answer'].append(('null', 0.00000, "null", 0.0000, 'null', 0.0000, question, query_tokens, 'null'))
        return result

    def entity_linking(self, question):
        query_tokens = self.get_mentions(question, self.questions_ent, self.model_ent, self.index2tag, self.args)
        query_tokens = query_tokens[0]  # default: choose the longest ngram as mention
        # entity linking
        cand_entity = []  # candidate entities
        cand_score = []  # add score
        ngrams_tokens = get_ngram(query_tokens.lower())  # bugfixed: lower()
        if len(ngrams_tokens) > 0:
            maxlen = len(ngrams_tokens[0].split())
        for item in ngrams_tokens:  # todo: maxlen
            if len(item.split()) < maxlen and len(cand_entity) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(cand_entity) > 0:
                break
            if item in stopwords:
                continue
            if item in self.index_ent:  # bugfixed: unk
                cand_entity.extend(self.index_ent[item])

        for (mid, text, typ) in sorted(set(cand_entity)):
            score = fuzz.ratio(text, query_tokens.lower()) / 100.0  # todo: may have bug
            # cand_entity_counts format : ((mid, text, type), score_based_on_fuzz)
            if mid in self.mid2url:
                # name = ";".join(self.mid2name[mid]) if mid in self.mid2name else "No name found!"
                # alias = ";".join(self.mid2alias[mid][:3]) if mid in self.mid2alias else "No alias found!"
                cand_score.append((mid, text, typ, score, query_tokens, question, self.mid2url[mid]))
            else:
                # name = ";".join(self.mid2name[mid]) if mid in self.mid2name else "No name found!"
                # alias = ";".join(self.mid2alias[mid][:3]) if mid in self.mid2alias else "No alias found!"
                cand_score.append((mid, text, typ, score, query_tokens, question, "no wiki entity found"))
        result = dict()
        result['answer'] = list()
        result['flag'] = True
        cand_score.sort(key=lambda t: t[3], reverse=True)
        # print("\ncandidate entity:")
        if len(cand_score) > 0:
            # print(cand_score[0])  # cand_score[:top_num]
            result['answer'] = cand_score
        else:
            result['flag'] = False
            result['answer'].append('null')
        return result

    def relation_detection(self, question):
        pred_relation = self.get_relations(question, self.questions_rel, self.model_rel, self.index2rel, self.args)
        result = dict()
        result['answer'] = list()
        result['flag'] = True
        pred_relation.sort(key=lambda t: t[1], reverse=True)   # rel, rel_log_score
        # print("\ncandidate relation:")
        if len(pred_relation) > 0:
            # print(pred_relation[0])  # cand_score[:top_num]
            for (rela, score) in pred_relation:
                result['answer'].append((rela, score, question))
        else:
            result['flag'] = False
            result['answer'].append('null')
        return result

    def mention_detection(self, questions):
        questions_list = questions.split("\n")
        result = dict()
        result['answer'] = list()
        result['flag'] = True

        for question in questions_list:
            query_tokens = self.get_mentions(question, self.questions_ent, self.model_ent, self.index2tag, self.args)
            query_tokens = query_tokens[0]  # default: choose the longest ngram as mention
            if len(query_tokens) > 0:
                result['answer'].append((query_tokens, question.replace("\r", "")))
        if len(result['answer']) == 0:
            result['flag'] = False
        # print(result)
        return result

    def mention_detection2(self, questions):
        questions_list = questions.split("\n")
        result = dict()
        result['answer'] = list()
        result['flag'] = True
        for question in questions_list:
            query_tokens = self.get_mentions(question, self.questions_ent, self.model_ent, self.index2tag, self.args)
            # query_tokens = query_tokens[0]
            if len(query_tokens) > 0:
                result['answer'].append(("|||".join(query_tokens), question.replace("\r", "")))
        if len(result['answer']) == 0:
            result['flag'] = False
        # print(result)
        return result
