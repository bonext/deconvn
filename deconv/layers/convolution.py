from ..tdeconv_utils import infer_conv_shapes

class Convolution(object):
    def __init__(self, f_shape_short, y_shape=None):
        self.kind = 'C'
        self.f_shape_short = f_shape_short
        self.y_shape = None
        self.z_shape = None
        self.f_shape = None
        if y_shape is not None:
            self.infer_shapes(y_shape)
        self.input_shape = self.y_shape
        self.output_shape = self.z_shape

    def infer_shapes(self, y_shape):
        self.y_shape = y_shape
        self.f_shape, self.z_shape = infer_conv_shapes(y_shape, self.f_shape_short)
        self.input_shape = self.y_shape
        self.output_shape = self.z_shape
