import numpy as np

import theano
import theano.tensor as T


class Model:
    def __init__(self):
        self.layers = []
        self.filters = []
        self.train_layers = []

    def add(self, layer):
        # populate self.filters
        # populate self.features ?
        pass

    def compile(self):
        # populate self.train_layers
        pass

    def fit(self, data, nb_epochs):
        switches = []
        for l in range(len(self.train_layers)):
            layer = self.train_layers[l]
            # init filters/features
            for epoch in range(1, nb_epochs+1):
                for y in data:
                    layer.fit_features(y, self.filters, switches)
                layer.fit_filters(self.filters, switches)
            switches.append(layer.get_switches())
            self.filters.append(layer.get_filters())

    def __call__(self, y):
        # inference
        switches = []
        features = []
        for l in range(len(self.train_layers)):
            layer = self.train_layers[l]
            # init filters/features
            layer.fit_features(y, self.filters, switches)
            switches.append(layer.get_switches())
            features.append(layer.get_features())
        return features


from tdeconv_utils import t_mk_conv, t_mk_conv_transpose
class Convolution:
    """
    Just a regular convolution with given filters
    """
    def __init__(self, filters_shape):
        """
        :param filters_shape: filter_channels x num_feature_maps x filter_h x filter_w
        """
        self.filters_shape = filters_shape
        
        # creating theano functions and stuff
        self.filters = theano.shared(np.zeros(self.filters_shape, dtype=theano.config.floatX))
        in4 = T.tensor4(name='conv_in', dtype=theano.config.floatX)
        f4 = T.tensor4(name='filters', dtype=theano.config.floatX)
        self.f_conv = theano.function(
            [in4],
            t_mk_conv(in4, f4),
            givens=[(f4, self.filters)]
            )
        self.f_t_conv = theano.function(
            [in4],
            t_mk_conv_transpose(in4, f4),
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
        self.pooling_shape = pooling_shape
        self.pool_ch, self.pool_w, self.pool_h = pooling_shape

    def P(self, layer_input):
        """
        Do max-abs pooling and update switches
        layer_input: 4d floatX theano tensor
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

