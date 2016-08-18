from theano import tensor as T
from theano.tensor.nnet.neighbours import images2neibs, neibs2images


"""

Convolution utilities

"""


def t_mk_conv(t_in, t_filters):
    t_batch_sz = t_in.shape[0]
    t_in_ch = t_in.shape[1]
    t_conv_out = T.nnet.conv2d(t_in, t_filters, border_mode='valid')
    return t_conv_out


def t_mk_conv_transpose(t_in, t_filters):
    t_in_ch = t_in.shape[1]
    t_batch_sz = t_in.shape[0]
    t_filters_shape = t_filters.shape
    t_fs_transposed = t_filters.dimshuffle(1, 0, 2, 3)
    t_conv_out = T.nnet.conv2d(t_in, t_fs_transposed, border_mode='full', filter_flip=False)
    return t_conv_out


"""

Pooling utilities

"""

def t_mk_pool_ready(t_pool_input, t_pool_shape):
    """
    Prepare pooling input
    :param t_pool_input: 4D theano tensor batch_sz x channels x height x width
    :param t_pool_shape: theano lvector pool_ch x pool_h x pool_w
    :return: aux. sizes and input reshaped for pooling
    """
    # sizes
    # input
    t_batch_sz = t_pool_input.shape[0]
    t_in_ch = t_pool_input.shape[1]
    t_in_h = t_pool_input.shape[2]
    t_in_w = t_pool_input.shape[3]
    # pooling
    t_pool_ch = t_pool_shape[0]
    t_pool_h = t_pool_shape[1]
    t_pool_w = t_pool_shape[2]
    # output
    t_out_ch = (t_in_ch + t_pool_ch - 1) // t_pool_ch
    t_out_h = (t_in_h + t_pool_h - 1) // t_pool_h
    t_out_w = (t_in_w + t_pool_w - 1) // t_pool_w

    # we will need to pad input (probably), so here's the padded shape:
    t_padded_ch = t_out_ch * t_pool_ch
    t_padded_h = t_out_h * t_pool_h
    t_padded_w = t_out_w * t_pool_w
    t_padded_pool_in_z = T.zeros(T.stack([t_batch_sz, t_padded_ch, t_padded_h, t_padded_w]))
    t_padded_pool_in = T.inc_subtensor(t_padded_pool_in_z[:t_batch_sz, :t_in_ch, :t_in_h, :t_in_w], t_pool_input)

    # below is all computed
    # spatial pooling
    t_sp_pooled = images2neibs(t_padded_pool_in, T.stack([t_pool_h, t_pool_w]))
    # spatial pooling output shape
    # has size (B * C * H/h * W/w) x (h*w)
    t_sp_pooled_dims = t_sp_pooled.shape
    # lines per channel
    # H*W / (h*w)
    t_lpc = (t_padded_h * t_padded_w) // (t_pool_h * t_pool_w)
    # shape to collect channels
    t_ch_pool_prep_dims_1 = T.stack([t_sp_pooled_dims[0] // t_lpc, t_lpc, t_sp_pooled_dims[1]])
    # preparing pooling by channels
    # reshape to collect channels in a separate dimension
    t_ch_pool_prep_1 = T.reshape(t_sp_pooled, t_ch_pool_prep_dims_1)
    t_ch_pool_prep_2 = T.shape_padleft(T.transpose(t_ch_pool_prep_1, [1, 0, 2]))
    # prepare for channel pooling
    t_ch_pool_dims = T.stack([t_pool_ch, t_ch_pool_prep_dims_1[-1]])
    t_pool_ready = images2neibs(t_ch_pool_prep_2, t_ch_pool_dims)
    return t_batch_sz, t_in_ch, t_in_h, t_in_w, t_out_ch, t_out_h, t_out_w, t_pool_ready


def t_mk_reshape_pooled(t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pooled_raw):
    """
    Reshape pooling to output shape
    :param t_batch_sz:
    :param t_out_ch:
    :param t_out_h:
    :param t_out_w:
    :param t_pooled_raw: output from argmax-pooling (vector)
    :return: 4D tensor t_batch_sz x t_out_ch x t_out_h x t_out_w
    """
    # lines per spatial block
    t_lpsb = t_out_ch * t_batch_sz
    t_pool_out_prep_dims_1 = T.stack([t_pooled_raw.shape[0] // t_lpsb, t_lpsb])
    t_pool_out_prep_1 = T.reshape(t_pooled_raw, t_pool_out_prep_dims_1).transpose()
    t_pool_out_prep_dims_2 = T.stack([t_batch_sz, t_out_ch, t_out_h, t_out_w])
    t_pool_out = T.reshape(t_pool_out_prep_1, t_pool_out_prep_dims_2)
    return t_pool_out


def t_mk_pooling(t_pool_input, t_pool_shape):
    """
    Make theano graph for max-abs-pooling
    :param t_pool_input: 4d tensor
    :param t_pool_shape: lvector with pooling shape c x h x w
    :return: pooled 4d tensor, switch locations
    """
    t_batch_sz, t_in_ch, t_in_h, t_in_w, t_out_ch, t_out_h, t_out_w, t_pool_ready = t_mk_pool_ready(t_pool_input, t_pool_shape)

    # argmax-pooling
    t_switches = T.argmax(abs(t_pool_ready), axis=1)
    t_rows = T.arange(t_pool_ready.shape[0])
    t_pooled_raw = t_pool_ready[t_rows, t_switches]

    t_pool_out = t_mk_reshape_pooled(t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pooled_raw)
    t_orig_shape = T.stack([t_batch_sz, t_in_ch, t_in_h, t_in_w])
    return t_pool_out, t_switches, t_orig_shape


def t_mk_fixed_pooling(t_pool_input, t_pool_shape, t_switches):
    """
    Make theano graph for pooling with known switches
    :param t_pool_input:
    :param t_pool_shape:
    :param t_switches:
    :return: pooled 4d tensor, input switches
    """
    t_batch_sz, t_in_ch, t_in_h, t_in_w, t_out_ch, t_out_h, t_out_w, t_pool_ready = t_mk_pool_ready(t_pool_input, t_pool_shape)

    t_rows = T.arange(t_pool_ready.shape[0])
    t_pooled_raw = t_pool_ready[t_rows, t_switches]

    t_pool_out = t_mk_reshape_pooled(t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pooled_raw)
    t_orig_shape = T.stack([t_batch_sz, t_in_ch, t_in_h, t_in_w])
    return t_pool_out, t_switches, t_orig_shape


def t_mk_unpooling(t_unpool_input, t_pool_shape, t_switches, t_orig_shape):
    """
    Make theano graph for unpooling with known switches
    :param t_unpool_input:
    :param t_pool_shape:
    :param t_switches:
    :return: Unpooled result
    """
    # sizes
    t_batch_sz = t_unpool_input.shape[0]
    # input
    t_in_ch = t_unpool_input.shape[1]
    t_in_h = t_unpool_input.shape[2]
    t_in_w = t_unpool_input.shape[3]
    # pooling
    t_pool_ch = t_pool_shape[0]
    t_pool_h = t_pool_shape[1]
    t_pool_w = t_pool_shape[2]
    # padded output
    t_out_ch_padded = t_in_ch * t_pool_ch
    t_out_h_padded = t_in_h * t_pool_h
    t_out_w_padded = t_in_w * t_pool_w
    # cropped output
    t_out_ch = t_orig_shape[1]
    t_out_h = t_orig_shape[2]
    t_out_w = t_orig_shape[3]

    # reshape input to get ready for unpool
    t_lpsb = t_batch_sz * t_in_ch
    t_pooled_raw = T.reshape(t_unpool_input, T.stack([t_lpsb, t_unpool_input.size//t_lpsb])).transpose().ravel()

    # ok, now unpool with switches
    t_raw_out_init = T.zeros(T.stack([t_batch_sz*t_in_ch*t_in_h*t_in_w, t_pool_ch*t_pool_h*t_pool_w]))
    t_rows = T.arange(t_raw_out_init.shape[0])
    t_raw_out = T.inc_subtensor(t_raw_out_init[t_rows, t_switches], t_pooled_raw)

    # now we only need to reshape back
    t_out_prep_shape_1 = T.stack([1, t_in_h * t_in_w, t_batch_sz * t_out_ch_padded, t_pool_h * t_pool_w])
    t_out_prep_1 = neibs2images(t_raw_out, T.stack([t_pool_ch, t_pool_h*t_pool_w]), t_out_prep_shape_1)
    t_out_prep_2 = T.transpose(t_out_prep_1[0], [1, 0, 2])
    t_out_prep_shape_3 = T.stack([t_batch_sz*t_out_ch_padded*t_in_h*t_in_w, t_pool_h*t_pool_w])
    t_out_prep_3 = T.reshape(t_out_prep_2, t_out_prep_shape_3)
    t_unpool_out_shape = T.stack([t_batch_sz, t_out_ch_padded, t_out_h_padded, t_out_w_padded])
    t_unpool_out_padded = neibs2images(t_out_prep_3, T.stack([t_pool_h, t_pool_w]), t_unpool_out_shape)
    # cut original data
    t_unpool_out = t_unpool_out_padded[:t_batch_sz, :t_out_ch, :t_out_h, :t_out_w]

    return t_unpool_out
