import networkx as nx
import scipy.sparse as sp

def load_data(dataset):
    """ Load datasets from text files

    :param dataset: 'cora', 'citeseer' or 'google' graph dataset.
    :return: n*n sparse adjacency matrix and n*f node-level feature matrix

    Note: in this paper, all three datasets are assumed to be featureless.
    As a consequence, the feature matrix is the identity matrix I_n.
    """
    if dataset == 'cora':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/cora.cites",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        # Transpose the adjacency matrix, as Cora raw dataset comes with a
        # <ID of cited paper> <ID of citing paper> edgelist format.
        adj = adj.T
        features = sp.identity(adj.shape[0])

    elif dataset == 'citeseer':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/citeseer.cites",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        # Transpose the adjacency matrix, as Citeseer raw dataset comes with a
        # <ID of cited paper> <ID of citing paper> edgelist format.
        adj = adj.T
        features = sp.identity(adj.shape[0])

    elif dataset == 'google':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/GoogleNw.txt",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        features = sp.identity(adj.shape[0])

    else:
        raise ValueError('Undefined dataset!')

    return adj, features