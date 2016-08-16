import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
floatX = theano.config.floatX

from deconv import Pooling

def t_mk_pooling(t_pool_input, t_sp_pool_dims, t_ch_pool_sz):
    # below is all computed
    # spatial pooling
    t_sp_pooled = images2neibs(t_pool_input, t_sp_pool_dims)
    # spatial pooling output shape
    # has size (B * C * H/h * W/w) x (h*w)
    t_sp_pooled_dims = t_sp_pooled.shape

    # lines per channel
    # H*W / (h*w)
    t_lpc = T.prod(t_pool_input.shape[-2:])//T.prod(t_sp_pool_dims)
    # shape to collect channels
    t_ch_pool_prep_dims_1 = T.stack([t_sp_pooled_dims[0]//t_lpc, t_lpc, t_sp_pooled_dims[1]])

    # preparing pooling by channels
    # reshape to collect channels in a separate dimension
    t_ch_pool_prep_1 = T.reshape(t_sp_pooled, t_ch_pool_prep_dims_1)
    t_ch_pool_prep_2 = T.shape_padleft(T.transpose(t_ch_pool_prep_1, [1, 0, 2]))

    # prepare for channel pooling
    t_ch_pool_dims = T.stack([t_ch_pool_sz, t_ch_pool_prep_dims_1[-1]])
    t_pool_ready = images2neibs(t_ch_pool_prep_2, t_ch_pool_dims)

    # argmax-pooling
    t_sel_cols = T.argmax(abs(t_pool_ready), axis=1)
    t_rows = T.arange(t_pool_ready.shape[0])
    t_pooled_raw = t_pool_ready[t_rows, t_sel_cols]

    # output channels
    t_out_ch = t_pool_input.shape[1]//t_ch_pool_sz
    t_out_h = t_pool_input.shape[2]//t_sp_pool_dims[0]
    t_out_w = t_pool_input.shape[3]//t_sp_pool_dims[1]

    # lines per spatial block
    t_lpsb = t_out_ch * t_pool_input.shape[0]
    t_pool_out_prep_dims_1 = T.stack([t_pooled_raw.shape[0]//t_lpsb, t_lpsb])
    t_pool_out_prep_1 = T.reshape(t_pooled_raw, t_pool_out_prep_dims_1).transpose()
    t_pool_out_prep_dims_2 = T.stack([t_pool_input.shape[0], t_out_ch, t_out_h, t_out_w])
    t_pool_out = T.reshape(t_pool_out_prep_1, t_pool_out_prep_dims_2)

    return (t_pool_out, t_sel_cols, t_pooled_raw)

if __name__ == "__main__":
    ## Check
    pool_in = np.random.randn(7, 6, 8, 12).astype(floatX)
    pool_in *= 100
    pool_in = np.round(pool_in)
    pool_in /= 100
    pool_sp_shape = np.array([2, 4], dtype=np.int64)
    ch_pool_sz = 3



    t_pool_input = T.tensor4(name='pool_input') # pool input batch x channels x H x W
    # spatial pooling dimensions
    # has size h x w
    t_sp_pool_dims = T.lvector(name='spatial_pool_dims')
    # channel pooling dimensions
    # scalar
    t_ch_pool_sz = T.lscalar(name='channel_pool_size')

    t_pool_out, t_sel_cols = t_mk_pooling(t_pool_input, t_sp_pool_dims, t_ch_pool_sz)
    pool_out = t_pool_out.eval({
            t_pool_input: pool_in,
            t_sp_pool_dims: pool_sp_shape,
            t_ch_pool_sz: ch_pool_sz
        })

    print pool_out.shape

    # P = Pooling(pool_in[0].shape, (ch_pool_sz,)+tuple(pool_sp_shape))
    #
    # for ix in range(pool_in.shape[0]):
    #     print np.all(pool_out[ix] == P.P(pool_in[ix]))