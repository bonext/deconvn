import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import neibs2images
floatX = theano.config.floatX

from deconv import Pooling

t_unpool_input = T.tensor4(name='pool_input')  # pool input batch x channels x H x W
# spatial pooling dimensions
t_sp_pool_dims = T.lvector(name='spatial_pool_dims')
# channel pooling dimensions
# scalar
t_ch_pool_sz = T.lscalar(name='channel_pool_size')
# unpool switches
t_sel_cols = T.lvector(name='unpool_switches')

def t_mk_unpool(t_unpool_input, t_sp_pool_dims, t_ch_pool_sz, t_switches):
    # sizes
    t_batch_sz = t_unpool_input.shape[0]
    # input
    t_in_ch = t_unpool_input.shape[1]
    t_in_h = t_unpool_input.shape[2]
    t_in_w = t_unpool_input.shape[3]
    # output
    t_out_ch = t_in_ch * t_ch_pool_sz
    t_out_h = t_in_h * t_sp_pool_dims[0]
    t_out_w = t_in_w * t_sp_pool_dims[1]
    # pooling
    t_pool_ch = t_ch_pool_sz
    t_pool_h = t_sp_pool_dims[0]
    t_pool_w = t_sp_pool_dims[1]


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

    return (t_unpool_out)