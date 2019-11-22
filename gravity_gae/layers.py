from __future__ import division
from gravity_gae.initializations import weight_variable_glorot
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
_LAYER_UIDS = {} # Global unique layer ID dictionary for layer name assignment

"""
Disclaimer: functions and classes defined from lines 16 to 122 in this file 
come from tkipf/gae original repository on Graph Autoencoders. Functions and 
classes from line 125 correspond to Source-Target and Gravity-Inspired 
decoders from our paper.
"""

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """ Graph convolution layer """
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs"""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Symmetric inner product decoder layer"""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class SourceTargetInnerProductDecoder(Layer):
    """Source-Target asymmetric decoder for directed link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(SourceTargetInnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        # Source vector = First half of embedding vector
        inputs_source = inputs[:, 0:int(FLAGS.dimension/2)]
        # Target vector = Second half of embedding vector
        inputs_target = inputs[:, int(FLAGS.dimension/2):FLAGS.dimension]
        # Source-Target decoding
        x = tf.matmul(inputs_source, inputs_target, transpose_b = True)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class GravityInspiredDecoder(Layer):
    """Gravity-Inspired asymmetric decoder for directed link prediction."""
    def __init__(self, normalize=False, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(GravityInspiredDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.normalize = normalize

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        # Embedding vector = all dimensions on input except the last
        # Mass parameter = last dimension of input
        if self.normalize:
            inputs_z = tf.math.l2_normalize(inputs[:,0:(FLAGS.dimension - 1)],
                                            axis = 1)
        else:
            inputs_z = inputs[:, 0:(FLAGS.dimension - 1)]
        # Get pairwise node distances in embedding
        dist = pairwise_distance(inputs_z, FLAGS.epsilon)
        # Get mass parameter
        inputs_mass = inputs[:,(FLAGS.dimension - 1):FLAGS.dimension]
        mass = tf.matmul(tf.ones([tf.shape(inputs_mass)[0],1]),tf.transpose(inputs_mass))
        # Gravity-Inspired decoding
        outputs = mass - tf.scalar_mul(FLAGS.lamb, tf.log(dist))
        outputs = tf.reshape(outputs,[-1])
        outputs = self.act(outputs)
        return outputs

def pairwise_distance(X, epsilon=0.01):
    """ Computes pairwise distances between node pairs
    :param X: n*d embedding matrix
    :param epsilon: add a small value to distances for numerical stability
    :return: n*n matrix of squared euclidean distances
    """
    x1 = tf.reduce_sum(X * X, 1, True)
    x2 = tf.matmul(X, tf.transpose(X))
    # Add epsilon to distances, avoiding 0 or too small distances leading to
    # numerical instability in gravity decoder due to logarithms
    return x1 - 2 * x2 + tf.transpose(x1) + epsilon