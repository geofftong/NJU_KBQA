# 75910/10845/21687
train_file = "../data/SimpleQuestions/output/simple.train.top20el.top300relation"
dev_file = "../data/SimpleQuestions/output/simple.valid.top20el.top300relation"
test_file = "../data/SimpleQuestions/output/simple.test.top20el.top300relation"
train_output_file = "../data/SimpleQuestions/output/simple.train.top20el.top100relation.score"
dev_output_file = "../data/SimpleQuestions/output/simple.valid.top20el.top100relation.score"
test_output_file = "../data/SimpleQuestions/output/simple.test.top20el.top300relation.score"
word2vec_file = "../data/word2vec/gigaxin_ldc_vectors.min5.en"
neg_sample = 50  # train 50
num_classes = 300  # 20 valid/test
max_sent_len = 36
model_name = "model"
voc_size = 288694
emb_size = 100
filter_sizes = [2, 3, 4]
num_filters = 50
dropout_keep_prob = 0.85  # 1.0
embeddings_trainable = True
epoch_num = 30  # 30 epoches
batch_size = 16  # 32
l2_reg_lambda = 0.1
early_step = 3  # epoch step of no improving in dev set

# lstm
hidden_units = 128
