from collections import OrderedDict

import numpy as np
from scipy.signal import convolve2d, correlate2d

class Pooling:
    def __init__(self, input_shape, pooling_shape):
        """
        Create a pooling operator
        :param input_shape:
        :param pooling_shape: pooling operator dimensions, must divide input dimensions
        """
        # check input shapes
        if not len(input_shape) == len(pooling_shape):
            errtxt = "Pooling operator shape must be same as pooling input\n"
            errtxt += "Got input_shape: %s pooling_shape: %s" % (input_shape, pooling_shape)
            raise RuntimeError(errtxt)
        if not reduce(lambda a, x: a and x, map(lambda z: z[0] % z[1] == 0, zip(input_shape, pooling_shape))) == True:
            errtxt = "Pooling operator shape must divide input shape\n"
            errtxt += "Got input_shape: %s pooling_shape: %s" % (input_shape, pooling_shape)
            raise RuntimeError(errtxt)
        if len(input_shape) != 3:
            raise NotImplementedError()

        self.input_shape = input_shape
        self.pooling_shape = pooling_shape
        self.output_shape = tuple(x/y for (x, y) in zip(input_shape, pooling_shape))
        self.argmax = None

    def P(self, layer_input):
        """
        Do (nan)max-abs pooling
        :param layer_input: numpy array of num_features x height x width, same shape as input_shape in c-tor
        :return: pooled array
        """
        out = np.zeros(self.output_shape, layer_input.dtype)
        self.argmax = np.zeros(self.output_shape, np.int32)
        p = self.pooling_shape
        for d in range(self.output_shape[0]):
            for y in range(self.output_shape[1]):
                for x in range(self.output_shape[2]):
                    chunk = layer_input[d*p[0]:(d+1)*p[0], y*p[1]:(y+1)*p[1], x*p[2]:(x+1)*p[2]]
                    amx = np.nanargmax(np.abs(chunk))
                    self.argmax[d, y, x] = amx
                    out[d, y, x] = chunk.flatten()[amx]
        return out

    def __call__(self, layer_input):
        """
        Do fixed-index pooling using pooling op state
        :param layer_input:
        :return:
        """
        if self.argmax is None:
            raise RuntimeError('Attempting to call fixed-index pooling before index was created')
        out = np.zeros(self.output_shape, layer_input.dtype)
        p = self.pooling_shape
        for d in range(self.output_shape[0]):
            for y in range(self.output_shape[1]):
                for x in range(self.output_shape[2]):
                    chunk = layer_input[d * p[0]:(d + 1) * p[0], y * p[1]:(y + 1) * p[1], x * p[2]:(x + 1) * p[2]]
                    out[d, y, x] = chunk.flatten()[self.argmax[d, y, x]]
        return out

    def T(self, layer_input):
        """
        Do transpose-pooling (fill in correct places with layer_input and set everything else to 0)
        :param layer_input: numpy array
        :return: transpose-pooled output
        """
        if self.argmax is None:
            raise RuntimeError('Attempting to call pooling transpose before pooling')
        if not layer_input.shape == self.output_shape:
            raise RuntimeError('Attempting to do transpose convolution with wrong input size')

        out = np.zeros(self.input_shape, layer_input.dtype)
        p = self.pooling_shape
        for d in range(self.output_shape[0]):
            for y in range(self.output_shape[1]):
                for x in range(self.output_shape[2]):
                    out[d * p[0]:(d + 1) * p[0], y * p[1]:(y + 1) * p[1], x * p[2]:(x + 1) * p[2]][np.unravel_index(self.argmax[d, y, x], p)] = layer_input[d, y, x]
        return out


class Convolution:
    """
    Just a regular convolution with given filters
    """
    def __init__(self, filters_shape):
        """
        :param filters_shape: filter_channels x num_feature_maps x filter_h x filter_w
        """
        self.filters_shape = filters_shape
        self.filter_channels, self.num_feature_maps, self.filter_h, self.filter_w = filters_shape
        self.filters = np.zeros(self.filters_shape)

    def __call__(self, features):
        """
        Convolve input features with filters
        :param features: numpy array num_features x H x W
        :return: non-padded convolution result
        """
        if len(features.shape) != 3:
            raise RuntimeError('Unexpected input dimensions')

        in_num_feats, in_feat_h, in_feat_w = features.shape

        if in_num_feats != self.num_feature_maps:
            raise RuntimeError('Number of input features is different from number of filters')

        yhat = np.zeros((self.filter_channels, in_feat_h + 1 - self.filter_h, in_feat_w + 1 - self.filter_w))
        for c in range(self.filter_channels):
            yhat[c] = np.sum(
                [convolve2d(features[k], self.filters[c][k], mode='valid') for k in range(self.num_feature_maps)],
                axis=0
            )
        return yhat

    def T(self, signal):
        """
        Transpose-convolution of signal with filters
        :param signal: numpy array num_channels x H x W
        :return: zero-padded full cross-correlation result
        """
        if len(signal.shape) != 3:
            raise RuntimeError('Unexpected input dimensions')

        in_num_channels, in_feat_h, in_feat_w = signal.shape

        if in_num_channels != self.filter_channels:
            raise RuntimeError('Number of signal channels is different from filter channels')

        features = np.zeros((self.num_feature_maps, in_feat_h + self.filter_h - 1, in_feat_w + self.filter_w - 1))
        for k in range(self.num_feature_maps):
            features[k] = np.sum(
                [correlate2d(signal[c], self.filters[c][k], 'full', 'fill', 0) for c in range(self.filter_channels)],
                axis=0
            )
        return features
