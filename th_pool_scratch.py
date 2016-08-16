import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

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
    t_out_ch = t_in_ch // t_pool_ch
    t_out_h = t_in_h // t_pool_h
    t_out_w = t_in_w // t_pool_w
    # below is all computed
    # spatial pooling
    t_sp_pooled = images2neibs(t_pool_input, T.stack([t_pool_h, t_pool_w]))
    # spatial pooling output shape
    # has size (B * C * H/h * W/w) x (h*w)
    t_sp_pooled_dims = t_sp_pooled.shape
    # lines per channel
    # H*W / (h*w)
    t_lpc = (t_in_h * t_in_w) // (t_pool_h * t_pool_w)
    # shape to collect channels
    t_ch_pool_prep_dims_1 = T.stack([t_sp_pooled_dims[0] // t_lpc, t_lpc, t_sp_pooled_dims[1]])
    # preparing pooling by channels
    # reshape to collect channels in a separate dimension
    t_ch_pool_prep_1 = T.reshape(t_sp_pooled, t_ch_pool_prep_dims_1)
    t_ch_pool_prep_2 = T.shape_padleft(T.transpose(t_ch_pool_prep_1, [1, 0, 2]))
    # prepare for channel pooling
    t_ch_pool_dims = T.stack([t_pool_ch, t_ch_pool_prep_dims_1[-1]])
    t_pool_ready = images2neibs(t_ch_pool_prep_2, t_ch_pool_dims)
    return t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pool_ready


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
    t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pool_ready = t_mk_pool_ready(t_pool_input, t_pool_shape)

    # argmax-pooling
    t_switches = T.argmax(abs(t_pool_ready), axis=1)
    t_rows = T.arange(t_pool_ready.shape[0])
    t_pooled_raw = t_pool_ready[t_rows, t_switches]

    t_pool_out = t_mk_reshape_pooled(t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pooled_raw)

    return (t_pool_out, t_switches)


def t_mk_fixed_pooling(t_pool_input, t_pool_shape, t_switches):
    """
    Make theano graph for pooling with known switches
    :param t_pool_input:
    :param t_pool_shape:
    :param t_switches:
    :return: pooled 4d tensor, input switches
    """
    t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pool_ready = t_mk_pool_ready(t_pool_input, t_pool_shape)

    t_rows = T.arange(t_pool_ready.shape[0])
    t_pooled_raw = t_pool_ready[t_rows, t_switches]

    t_pool_out = t_mk_reshape_pooled(t_batch_sz, t_out_ch, t_out_h, t_out_w, t_pooled_raw)

    return (t_pool_out, t_switches)