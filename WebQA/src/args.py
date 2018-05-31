import os

from argparse import ArgumentParser


def get_kb_args():
    parser = ArgumentParser(description='Simple QA model: knowledge base')
    parser.add_argument('--idx_entity', type=str, default='data/indexes/entity_2M.pkl')
    parser.add_argument('--idx_reachability', type=str, default='data/indexes/reachability_2M.pkl')
    parser.add_argument('--idx_name', type=str, default='data/indexes/names_2M.pkl')
    parser.add_argument('--idx_freebase', type=str, default='data/indexes/fb_graph.pkl')
    parser.add_argument('--idx_degree', type=str, default='data/indexes/degrees_2M.pkl')
    parser.add_argument('--wiki_path', type=str, default='data/fb2w.nt')
    parser.add_argument('--_path', type=str, default='data/fb2w.nt')
    parser.add_argument('--name_path', type=str, default='data/indexes/names_only_2M.pkl')
    parser.add_argument('--alias_path', type=str, default='data/indexes/alias_only_2M.pkl')

    args = parser.parse_args()
    return args


def get_entity_args():
    parser = ArgumentParser(description='Simple QA model: mention detection')
    parser.add_argument('--results_path', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--trained_model', type=str, default='model/mention_detection/lstm_id1_model_cpu.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dim_hidden', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true', dest='test', help='get the testing set result')
    parser.add_argument('--rnn_type', type=str, default='lstm')  # or use 'gru'
    parser.add_argument('--dim_embed', type=int, default=300)
    parser.add_argument('--dev', action='store_true', dest='dev', help='get the development set result')
    parser.add_argument('--not_bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--dev_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=4500)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=5, help="number of epochs to wait before early stopping")
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device to use')  # use -1 for CPU
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/processed_dataset')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), 'data/cache'))  # ?
    parser.add_argument('--vector_cache', type=str,
                        default=os.path.join(os.getcwd(), 'data/cache/sq_glove300d.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')  # fine-tune the word embeddings
    parser.add_argument('--resume_snapshot', type=str, default=None)

    args = parser.parse_args()
    return args


def get_relation_args():
    parser = ArgumentParser(description='Simple QA model: relation detection')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_type', type=str, default='gru', help="use 'gru' or 'lstm'")
    parser.add_argument('--dim_embed', type=int, default=300)
    parser.add_argument('--dim_hidden', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_weight_decay', type=float, default=0.0)
    # parser.add_argument('--patience', type=int, default=3, help="number of epochs to wait before early stopping")
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--not_bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device to use')  # use -1 for CPU
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--data_dir', type=str, default='data/processed_dataset')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), 'data/cache'))
    parser.add_argument('--vector_cache', type=str,
                        default=os.path.join(os.getcwd(), 'data/cache/sq_glove300d.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')  # fine-tune the word embeddings
    parser.add_argument('--trained_model', type=str, default='model/relation_detection/cnn_id1_model_cpu.pt')
    parser.add_argument('--hits', type=int, default=1000, help="number of top results to output")

    args = parser.parse_args()
    return args



