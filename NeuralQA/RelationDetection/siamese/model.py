import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, cuda, mem_dim):
        super(Attention, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.WQ = nn.Linear(mem_dim, mem_dim)  # bias=False
        # self.WV = nn.Linear(mem_dim, mem_dim)
        # self.WP = nn.Linear(mem_dim, 1)

    def forward(self, query_h, doc_h):
        # doc_h = torch.unsqueeze(doc_h, 0)
        # ha = F.tanh(self.WQ(query_h) + self.WV(doc_h).expand_as(query_h))  # tan(W1a + W2b)
        # p = F.softmax(self.WP(ha).squeeze())
        # weighted = p.unsqueeze(1).expand_as(
        #     query_h) * query_h
        # v = weighted.sum(dim=0)
        p = F.softmax(torch.transpose(torch.mm(query_h, doc_h.unsqueeze(1)), 0, 1))  # dot
        weighted = torch.transpose(p, 0, 1).expand_as(query_h) * query_h
        v = weighted.sum(dim=0)
        return v, p


class RelationDetection(nn.Module):
    def __init__(self, config):
        super(RelationDetection, self).__init__()
        self.config = config
        target_size = config.rel_label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        # self.attention = Attention(cuda, mem_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        if config.relation_detection_mode.upper() == "GRU":
            self.gru = nn.GRU(input_size=config.input_size,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_layer,
                              dropout=config.rnn_dropout,
                              bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_detection_mode.upper() == "LSTM":
            self.lstm = nn.LSTM(input_size=config.input_size,
                                hidden_size=config.hidden_size,
                                num_layers=config.num_layer,
                                dropout=config.rnn_dropout,
                                bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_detection_mode.upper() == "CNN":
            input_channel = 1
            Ks = 3
            self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
            self.dropout = nn.Dropout(config.cnn_dropout)
            self.fc1 = nn.Linear(Ks * config.output_channel, target_size)

    def forward(self, ques, rela_list):
        batch_size = ques.size()[0]
        ques = self.embed(ques)  # (batch_size, sent_len, embed_dim)
        rela_list = [self.embed(rela) for rela in rela_list]  # (num_classes, batch_size, sent_len, embed_dim)
        rela_output = list()
        if self.config.relation_detection_mode.upper() == "LSTM":
            # h0 / c0 = (layer*direction, batch_size, hidden_dim)
            if self.config.cuda:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
                c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
                c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
            # output = (sentence length, batch_size, hidden_size * num_direction)
            # ht = (layer*direction, batch, hidden_dim)
            # ct = (layer*direction, batch, hidden_dim)
            outputs1, (ht1, ct1) = self.lstm(ques, (h0, c0))
            # cross attention

            # query_cross_alphas = Var(torch.Tensor(query_state.size(0), target_state.size(0)))
            # target_cross_alphas = Var(torch.Tensor(target_state.size(0), query_state.size(0)))
            # q_to_t = Var(torch.Tensor(query_state.size(0), self.mem_dim))
            # t_to_q = Var(torch.Tensor(target_state.size(0), self.mem_dim))
            # for rela in rela_list:
            #     outputs2, (ht2, ct2) = self.lstm(rela, (h0, c0))
            #     for i in range(query_state.size(0)):
            #         q_to_t[i], query_cross_alphas[i] = self.attention(target_state, query_state[i,])
            tags = self.hidden2tag(ht1[-2:].transpose(0, 1).contiguous().view(batch_size, -1))
            scores = F.log_softmax(tags)
            return scores
        elif self.config.relation_detection_mode.upper() == "GRU":
            if self.config.cuda:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size))
            outputs, ht = self.gru(ques, h0)

            tags = self.hidden2tag(ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1))
            scores = F.log_softmax(tags)
            return scores
        elif self.config.relation_detection_mode.upper() == "CNN":
            ques = ques.contiguous().unsqueeze(1)
            ques = [F.relu(self.conv1(ques)).squeeze(3), F.relu(self.conv2(ques)).squeeze(3),
                    F.relu(self.conv3(ques)).squeeze(3)]
            ques = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in ques]  # max-over-time pooling
            ques = torch.cat(ques, 1)  # (batch, channel_output * Ks)
            ques = self.dropout(ques)
            # logit = self.fc1(ques)  # (batch, target_size)
            ques = ques.unsqueeze(1)  # (batch, 1, channel_output * Ks)
            for rela in rela_list:
                rela = rela.contiguous().unsqueeze(1)  # rela.transpose(0, 1)
                rela = [F.relu(self.conv1(rela)).squeeze(3), F.relu(self.conv2(rela)).squeeze(3),
                        F.relu(self.conv3(rela)).squeeze(3)]
                rela = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in rela]
                rela = torch.cat(rela, 1)
                rela = self.dropout(rela)
                rela = rela.unsqueeze(1)
                rela_output.append(rela)
            rela = torch.cat(rela_output, 1).transpose(0, 1).contiguous()
            dot = torch.sum(torch.mul(ques, rela), 2)
            sqrt_ques = torch.sqrt(torch.sum(torch.pow(ques, 2), 2))
            sqrt_rela = torch.sqrt(torch.sum(torch.pow(rela, 2), 2))
            # print(sqrt_ques, sqrt_rela)  # 32,1  32,51
            epsilon = 1e-6  # 1e-6
            scores = dot / (sqrt_ques * sqrt_rela + epsilon)  # torch.max(a, b)???
            return scores
        else:
            print("Unknown Mode")
            exit(1)
