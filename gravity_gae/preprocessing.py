import numpy as np
import scipy.sparse as sp
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

"""
Disclaimer: functions defined from lines 15 to 36 in this file come from 
tkipf/gae original repository on Graph Autoencoders. Moreover, the
mask_test_edges_general_link_prediction function is borrowed from 
philipjackson's mask_test_edges pull request on this same repository.
"""

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj (see section 3.3.1 of paper)
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


# Edge Masking for the three directed link prediction tasks

def mask_test_edges_general_link_prediction(adj, test_percent=10., val_percent=5.):
    """
    Task 1: General Directed Link Prediction: get Train/Validation/Test

    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """

    # Remove diagonal elements of adjacency matrix
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape = adj.shape)
    adj.eliminate_zeros()
    edges_positive, _, _ = sparse_to_tuple(adj)

    # Number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # Sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    # positive val edges
    val_edges = edges_positive[val_edge_idx]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    # positive test edges
    test_edges = edges_positive[test_edge_idx]
    # positive train edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0)

    # (Text from philipjackson)
    # The above strategy for sampling without replacement will not work for sampling
    # negative edges on large graphs, because the pool of negative edges
    # is much much larger due to sparsity, therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll probably
    # have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. remove any duplicate elements if there are any
    # 5. remove any diagonal elements
    # 6. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices
    # Test set
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    # Validation set
    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # Sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], test_edges_linear))

    # Re-build train adjacency matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape = adj.shape)

    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_biased_negative_samples(adj, test_percent=10.):
    """
    Task 2: General Biased Negative Samples (B.N.S.) Directed Link
    Prediction: get Train and Test sets

    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :return: train incomplete adjacency matrix and test sets
    """

    # Remove diagonal elements of adjacency matrix
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape = adj.shape)
    adj.eliminate_zeros()
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    ## Retrieve all unidirectional edges
    adj_sym = (adj + adj.T).sign()
    adj_tilde = (adj_sym - adj).T
    adj_tilde.eliminate_zeros()
    edges_positive, _, _ = sparse_to_tuple(adj_tilde)

    # Number of positive (= to number of negative) test node pairs to sample
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))

    # Sampling of positive node pairs
    edges_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_idx)
    test_edges_idx = edges_idx[:num_test]
    test_edges = edges_positive[test_edges_idx]

    # In this setting, the reverse node pairs constitute negative samples
    test_edges_false = np.fliplr(test_edges)

    # Get training incomplete adjacency matrix
    train_edges = np.delete(edges_positive, np.hstack([test_edges_idx]), axis = 0)
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape = adj.shape)

    # Validation set: not implemented for Task 2
    # therefore, val_edges and val_edges_false are None
    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false


# Task 3 - Bidirectional Link Prediction
def mask_test_edges_bidirectional_link_prediction(adj):
    """
    Task 3: Bidirectionality Prediction: get Train and Test sets

    :param adj: complete sparse adjacency matrix of the graph
    :return: train incomplete adjacency matrix and test sets
    """

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape = adj.shape)
    adj.eliminate_zeros()
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    ## Unidirectional edges
    adj_sym = (adj + adj.T).sign()
    adj_tilde = (adj_sym - adj).T
    adj_tilde.eliminate_zeros()

    ## Bidirectional edges (they usually are few, so they are all in test set)
    adj_sym_ones = adj - adj_tilde
    adj_sym_ones.eliminate_zeros()
    if FLAGS.verbose:
        print('Number of bidirectional edges in the graph:',
              np.count_nonzero(np.asarray(adj_sym_ones.todense()))/2)

    # Positive node pairs in test set (bidirectional edges)
    test_edges, _, _ = sparse_to_tuple(adj_sym_ones)
    test_edges = test_edges[test_edges[:,1] > test_edges[:,0],:]

    # Negative node pairs in test set (unidirectional edges)
    test_edges_false, _, _ = sparse_to_tuple(adj_tilde)
    test_edges_false = test_edges_false[test_edges_false[:,0] > test_edges_false[:,1],:]
    test_edges_false = np.fliplr(test_edges_false)
    # Sampling of negative node pairs
    edges_negative_idx = np.arange(test_edges_false.shape[0])
    np.random.shuffle(edges_negative_idx)
    test_edges_false_idx = edges_negative_idx[:test_edges.shape[0]]
    test_edges_false = test_edges_false[test_edges_false_idx]

    # Get training incomplete adjacency matrix
    # 1 of the 2 directions of each bidirectional edge is masked
    adj_train = (adj - sp.triu(adj_sym_ones))

    # Validation set: not implemented for Task 2
    # val_edges and val_edges_false are None
    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false