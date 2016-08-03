import numpy as np

import theano
import theano.tensor as T

class Convolution:
    """
    Just a regular convolution with given filters
    """
    def __init__(self, filters_shape):
        """
        :param filters_shape: filter_channels x num_feature_maps x filter_h x filter_w
        """
        self.filters_shape = filters_shape
        self.t_filters_shape = filters_shape[2:0:-1]+filters_shape[2:]
        self.filter_channels, self.num_feature_maps, self.filter_h, self.filter_w = filters_shape
        
        # creating theano functions and stuff
        self.filters = theano.shared(np.zeros(self.filters_shape, dtype=theano.config.floatX))
        in4 = T.tensor4(dtype=theano.config.floatX)
        f4 = T.tensor4(dtype=theano.config.floatX)
        self.f_conv = theano.function(
            [in4],
            T.nnet.conv2d(in4, f4, (1, self.num_feature_maps, None, None), filters_shape, 'valid'),
            givens=[(f4, self.filters)]
            )
        self.f_t_conv = theano.function(
            [in4],
            T.nnet.conv2d(in4, f4.dimshuffle(1,0,2,3), (1, self.filter_channels, None, None), self.t_filters_shape, 'full', filter_flip=False),
            givens=[(f4, self.filters)]
            )


    def __call__(self, features):
        """
        Convolve input features with filters
        :param features: numpy array batch_size x num_features x H x W
        :return: non-padded convolution result
        """
        if len(features.shape) != 4:
            raise RuntimeError('Unexpected input dimensions')

        return self.f_conv(features)       

    def T(self, signal):
        """
        Transpose-convolution of signal with filters
        :param signal: numpy array batch_size x num_channels x H x W
        :return: zero-padded full cross-correlation result
        """
        if len(signal.shape) != 4:
            raise RuntimeError('Unexpected input dimensions')

        return self.f_t_conv(signal)


class Pooling:
    def __init__(self, pooling_shape):
        pass
        
    def P(self, layer_input):
        """
        Do max-abs pooling and update switches
        """
        pass

    def __call__(self, layer_input):
        """
        Use switches to select values
        """
        pass

    def T(self, layer_input):
        """
        Do transpose pooling using updated switches
        """
        pass

