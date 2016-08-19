from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

class TestTheanoConvolutions(object):
    def setUp(self):
        self.filters = np.array(
            [[[[1, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]],
             [[[0, 0, 0],
               [0, 0, 0],
               [0, 0, -1]]]
             ],
            dtype=theano.config.floatX
        )

    def test_theano_convolution(self):
        # how to use t_mk_conv
        from deconv.tdeconv_utils import t_mk_conv
        in4 = T.tensor4(name='conv_in', dtype=theano.config.floatX)
        f4 = T.tensor4(name='filters', dtype=theano.config.floatX)
        f_conv = theano.function(
            [in4],
            t_mk_conv(in4, f4),
            givens=[(f4, self.filters)]
        )

        test_input = np.array(
            [[[[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]]],
            dtype=theano.config.floatX
        )
        ground_truth = np.array(
            [[[[1,  1,  0],
               [1,  0,  0],
               [0,  0,  0]],
              [[0,  0,  0],
               [0,  0, -1],
               [0, -1, -1]]]],
            dtype=theano.config.floatX
        )
        assert_true(np.all(f_conv(test_input) == ground_truth))

    def test_theano_transposed_convolution(self):
        # how to use t_mk_conv_transpose
        from deconv.tdeconv_utils import t_mk_conv_transpose
        in4 = T.tensor4(name='conv_in', dtype=theano.config.floatX)
        f4 = T.tensor4(name='filters', dtype=theano.config.floatX)
        f_t_conv = theano.function(
            [in4],
            t_mk_conv_transpose(in4, f4),
            givens=[(f4, self.filters)]
        )

        test_input = np.array(
            [[[[0, 1, 0],
               [0, 1, 0],
               [0, 1, 0]],
              [[0, 0, 0],
               [1, 1, 1],
               [0, 0, 0]]]],
            dtype=theano.config.floatX
        )
        ground_truth = np.array(
            [[[[ 0,  0,  0,  0,  0],
               [-1, -1, -1,  0,  0],
               [ 0,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0],
               [ 0,  0,  0,  1,  0]]]],
            dtype=theano.config.floatX
        )
        assert_true(np.all(f_t_conv(test_input) == ground_truth))
