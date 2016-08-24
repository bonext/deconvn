from ..tdeconv_utils import infer_pool_shape

class Pooling(object):
    def __init__(self, p_shape, input_shape=None):
        self.kind = 'P'
        self.p_shape = p_shape
        self.input_shape = None
        self.output_shape = None
        if input_shape is not None:
            self.infer_shapes(input_shape)

    def infer_shapes(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = infer_pool_shape(input_shape, self.p_shape)
