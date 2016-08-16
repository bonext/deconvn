import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import neibs2images

def t_mk_unpool(t_unpool_input, t_pool_shape, t_switches):
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
    # output
    t_out_ch = t_in_ch * t_pool_ch
    t_out_h = t_in_h * t_pool_h
    t_out_w = t_in_w * t_pool_w

    # reshape input to get ready for unpool
    t_lpsb = t_batch_sz * t_in_ch
    t_pooled_raw = T.reshape(t_unpool_input, T.stack([t_lpsb, t_unpool_input.size//t_lpsb])).transpose().ravel()

    # ok, now unpool with switches
    t_raw_out_init = T.zeros(T.stack([t_batch_sz*t_in_ch*t_in_h*t_in_w, t_pool_ch*t_pool_h*t_pool_w]))
    t_rows = T.arange(t_raw_out_init.shape[0])
    t_raw_out = T.inc_subtensor(t_raw_out_init[t_rows, t_switches], t_pooled_raw)

    # now we only need to reshape back
    t_out_prep_shape_1 = T.stack([1, t_in_h * t_in_w, t_batch_sz * t_out_ch, t_pool_h * t_pool_w])
    t_out_prep_1 = neibs2images(t_raw_out, T.stack([t_pool_ch, t_pool_h*t_pool_w]), t_out_prep_shape_1)
    t_out_prep_2 = T.transpose(t_out_prep_1[0], [1, 0, 2])
    t_out_prep_shape_3 = T.stack([t_batch_sz*t_out_ch*t_in_h*t_in_w, t_pool_h*t_pool_w])
    t_out_prep_3 = T.reshape(t_out_prep_2, t_out_prep_shape_3)
    t_unpool_out_shape = T.stack([t_batch_sz, t_out_ch, t_out_h, t_out_w])
    t_unpool_out = neibs2images(t_out_prep_3, T.stack([t_pool_h, t_pool_w]), t_unpool_out_shape)

    return t_unpool_out