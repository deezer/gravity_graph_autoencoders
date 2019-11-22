from __future__ import division
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))

def compute_scores(edges_pos, edges_neg, emb):
    """ Computes AUC ROC and AP scores from embeddings vectors, and from
    ground-truth lists of positive and negative node pairs

    :param edges_pos: list of positive node pairs
    :param edges_neg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """
    dim = FLAGS.dimension # Embedding dimension
    epsilon = FLAGS.epsilon # For numerical stability (see layers.py file)
    preds = []
    preds_neg = []

    # Standard Graph AE/VAE
    if FLAGS.model in ('gcn_ae', 'gcn_vae'):
        for e in edges_pos:
            # Link Prediction on positive pairs
            preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
        for e in edges_neg:
            # Link Prediction on negative pairs
            preds_neg.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))

    # Source-Target Graph AE/VAE
    elif FLAGS.model in ('source_target_gcn_ae', 'source_target_gcn_vae'):
        for e in edges_pos:
            # Link Prediction on positive pairs
            preds.append(sigmoid(emb[e[0],0:int(dim/2)].dot(emb[e[1],int(dim/2):dim].T)))
        for e in edges_neg:
            # Link Prediction on negative pairs
            preds_neg.append(sigmoid(emb[e[0],0:int(dim/2)].dot(emb[e[1],int(dim/2):dim].T)))

    # Gravity-Inspired Graph AE/VAE
    elif FLAGS.model in ('gravity_gcn_ae', 'gravity_gcn_vae'):
        for e in edges_pos:
            # Link Prediction on positive pairs
            dist = np.square(epsilon +
                             np.linalg.norm(emb[e[0],0:(dim-1)]
                                            - emb[e[1],0:(dim-1)],ord=2))
            # Prediction = sigmoid(mass - lambda*log(distance))
            preds.append(sigmoid(emb[e[1],dim-1] - FLAGS.lamb*np.log(dist)))
        for e in edges_neg:
            # Link Prediction on negative pairs
            dist = np.square(epsilon +
                             np.linalg.norm(emb[e[0],0:(dim-1)]
                                            - emb[e[1],0:(dim-1)],ord=2))
            preds_neg.append(sigmoid(emb[e[1],dim-1] - FLAGS.lamb*np.log(dist)))

    # Stack all predictions and labels
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    # Computes metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score