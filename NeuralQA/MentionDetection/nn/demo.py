import os
import random

import numpy as np
import torch
from args import get_args
from datasets import SimpleQuestionDataset
from torchtext import data

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

np.set_printoptions(threshold=np.nan)
# Set default configuration in : args.py
args = get_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print(args)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    # print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

TEXT = data.Field(lower=True)
ED = data.Field()

train, dev, test = SimpleQuestionDataset.splits(TEXT, ED, path=args.data_dir)  # text_field, label_field
TEXT.build_vocab(train, dev, test)
ED.build_vocab(train, dev, test)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))

# print(model)

if args.dataset == 'EntityDetection':
    index2tag = np.array(ED.vocab.itos)
    # print(index2tag)
else:
    print("Wrong Dataset")
    exit(1)

index2word = np.array(TEXT.vocab.itos)

results_path = os.path.join(args.results_path, args.entity_detection_mode.lower())
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

sentence = "who wrote the film Brave heart"  # args.input which genre of album is harder ... faster
# sentence = args.input
# print(sentence.split())
# print(index2word[:10])  # '<unk>', '<pad>'

sent_idx = list()
for word in sentence.split():
    if word.lower() in index2word:
        sent_idx.append(index2word.tolist().index(word.lower()))
    else:
        sent_idx.append(0)  # 1?


# print(sent_idx)


def predict(data_name="demo"):
    model.eval()

    data_batch = np.array(sent_idx).reshape(-1, 1)
    scores = model(data_batch)
    # print(scores)

    if args.dataset == 'EntityDetection':
        # print(torch.max(scores, 1)[1])
        index_tag = np.transpose(torch.max(scores, 1)[1].cpu().data.numpy())
        tag_array = index2tag[index_tag]
        index_question = np.transpose(data_batch).reshape(-1)
        question_array = index2word[index_question]

        mentions = list()
        mention = ""
        for question, label in zip(question_array, tag_array):
            # print("{}\t{}\t".format("".join(question), " ".join(label)))
            if label == 'I' and question != "<unk>" and question != "<pad>":
                mention += question + " "
            if label == 'O' and mention != "":
                mentions.append(mention.strip())
                mention = ""
        if mention != "":
            mentions.append(mention.strip())
        for mention in mentions:
            print(mention)
    else:
        print("Wrong Dataset")
        exit()


# run the model on the demo set and write the output to a file
predict(data_name="demo")
