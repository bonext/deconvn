import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
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

def t_mk_unpool(t_unpool_input, t_sp_pool_dims, t_ch_pool_sz, t_sel_cols):
    # batch size
    t_batch_sz = t_unpool_input.shape[0]
    t_in_ch = t_unpool_input.shape[1]
    t_lpsb = t_batch_sz * t_in_ch
    t_pooled_raw = T.reshape(t_unpool_input, T.stack([t_lpsb, t_unpool_input.size//t_lpsb])).transpose().ravel()
    return (t_pooled_raw)