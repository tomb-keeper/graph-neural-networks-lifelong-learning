import torch
import torch_geometric


def add_node2vec_args(parser):
    """ Adds node2vec specific command line arguments to a given parser """
    parser.add_argument('--n2v_walk_length', type=int, default=20,
                        help="Length of node2vec random walks")
    parser.add_argument('--n2v_walks_per_node', type=int, default=10,
                        help="Number of walks per node")
    parser.add_argument('--n2v_num_negative_samples', type=int, default=1,
  