import torch
import torch_geometric


def add_node2vec_args(parser):
    """ Adds node2vec specific command line arguments to a given parser """
    parser.add_argument('--n2v_walk_length', type=int, default=20,
                        help="Length of node2vec random walks")
    parser.add_argument('--n2v_walks_per_node', type=int, default=10,
                        help="Number of walks per node")
    parser.add_argument('--n2v_num_negative_samples', type=int, default=1,
                        help="Num negative examples per positive example")
    parser.add_argument('--n2v_context_size', type=int, default=10,
                        help="Size of sliding window within random walks")
    parser.add_argument('--n2v_p', type=float, default=1.,
                        help="Node2vec p parameter")
    parser.add_argument('--n2v_q', type=float, default=1.,
                        help="Node2vec q parameter")
    parser.add_argument('--n2v_batch_size', type=int, default=128,
                        help="Node2vec batch size")
    parser.add_argument('--n2v_num_workers', type=int, default=4,
                        help="Node2vec workers (#threads)")

def train_node2vec(model, optimizer, epochs=1,
  