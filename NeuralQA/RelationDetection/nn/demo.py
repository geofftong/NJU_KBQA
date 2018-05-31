import os
import random
import torch
import numpy as np

from datasets import SimpleQuestionsDataset
from torchtext import data
from args import get_args


np.set_printoptions(threshold=np.nan)
# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    # print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

TEXT = data.Field(lower=True)
RELATION = data.Field(sequential=False)

train, dev, test = SimpleQuestionsDataset.splits(TEXT, RELATION, args.data_dir)
TEXT.build_vocab(train, dev, test)
RELATION.build_vocab(train, dev)

# load the model
model = torch.load(args.trained_model)
# print(model)

if args.dataset == 'RelationDetection':
    index2tag = np.array(RELATION.vocab.itos)
    print(len(index2tag))
    # print(index2tag[:5])
else:
    print("Wrong Dataset")
    exit(1)

index2word = np.array(TEXT.vocab.itos).tolist()
# print(len(index2word))

results_path = os.path.join(args.results_path, args.relation_detection_mode.lower())
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

sentence = "who is the child of Obama"  # args.input  obama
print(model.embed(torch.LongTensor([12])))
# sentence = args.input
# print(sentence.split())
# print(index2word.index('what'))
# print(index2word[:10])  # '<unk>', '<pad>'


sent_idx = list()
for word in sentence.split():
    if word.lower() in index2word:
        sent_idx.append(index2word.index(word.lower()))
    else:
        sent_idx.append(0)  # 1?
print(sent_idx)


def predict(data_name="test"):
    # print("Dataset: {}".format(data_name))
    model.eval()

    data_batch = np.array(sent_idx).reshape(-1, 1)
    # print(data_batch)

    scores = model(data_batch)  #
    # print(scores)

    if args.dataset == 'RelationDetection':
        # Get top k
        top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True)  # shape: (batch_size, k)
        top_k_scores_array = top_k_scores.cpu().data.numpy()
        top_k_indices_array = top_k_indices.cpu().data.numpy()
        top_k_relatons_array = index2tag[top_k_indices_array]
        for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
            for j, (rel, score) in enumerate(zip(relations_row, scores_row)):
                print("{}\t{}".format(rel, score))
    else:
        print("Wrong Dataset")
        exit()


# run the model on the test set and write the output to a file
predict(data_name="demo")
