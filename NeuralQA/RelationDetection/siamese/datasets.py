import util


class Dataset(object):
    def __init__(self, ques_list, rela_list, label, max_sent_len, word_dict):
        self.ques = ques_list  # [[]] ?
        self.rela = rela_list
        self.word_dict = word_dict
        self.label = label
        self.size = len(label)
        self.max_sent_len = max_sent_len
        self.ques_idx, self.rela_idx = self.get_voc_idx(self.ques, self.rela)

    def get_voc_idx(self, ques, rela):
        # pad sentence
        pad = lambda x: util.pad_sentences(x, self.max_sent_len)
        pad_lst = lambda x: list(map(pad, x))
        self.ques_pad = list(map(pad, ques))
        self.rela_pad = list(map(pad_lst, rela))
        # Represent sentences as list(nparray) of ints
        idx_func = lambda word: self.word_dict[word] if word in self.word_dict else self.word_dict["<pad>"]
        u_idx_func = lambda words: list(map(idx_func, words))
        v_idx_func = lambda words_list: list(map(u_idx_func, words_list))
        return list(map(u_idx_func, self.ques_pad)), list(map(v_idx_func, self.rela_pad))
