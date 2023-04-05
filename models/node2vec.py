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
                     