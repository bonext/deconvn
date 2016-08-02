import numpy as np

import theano
from theano.tensor.nnet.neighbours import images2neibs
import theano.tensor as T
floatX = theano.config.floatX



input = np.random.rand(2, 2, 4, 6).astype(theano.config.floatX)
in_shape = input.shape
in_batch_sz, in_channels, in_h, in_w = input.shape

# lets do it all in one step
pool_shape = [2, 2, 3] # channels x h x w
pool_ch, pool_h, pool_w = pool_shape

# Prepare all shapes
# images2neibs creates an array of size
# something x pool_h*pool_w
sp_pool_w = pool_h * pool_w
sp_pool_h = input.size / sp_pool_w

# spatial lines per channel - can be prepared in advance
lpc = np.prod(input.shape[-2:])/np.prod(pool_shape[-2:])

# reshape to this for channel pooling
ch_pool_shape = (sp_pool_h/lpc, lpc, sp_pool_w) # (sp_pool.shape[0]/lpc, lpc, sp_pool.shape[-1])

# shared variables
s_pool_sp_shape = theano.shared(np.array([pool_h, pool_w], dtype=np.int64))
s_pool_ch_shape = theano.shared(np.array([pool_ch, sp_pool_w], dtype=np.int64))
s_grouped_chs_shape = theano.shared(np.array(ch_pool_shape, dtype=np.int64))

# theano variables
th_pool_in = T.tensor4() # pool input batch x channels x H x W
th_pool_sp_shape = T.lvector() # spatial pooling shape
th_pool_ch_shape = T.lvector() # channel pooling shape
th_grouped_chs_shape = T.lvector() # shape ready for channel pooling

# building theano expression for pooling prep
# collect spatial neighbourhoods
sp_pool = images2neibs(th_pool_in, th_pool_sp_shape)

# collect channels in a separate dimension
sp_collect_chans = T.reshape(sp_pool, th_grouped_chs_shape, ndim=3)
next_in = T.shape_padleft(T.transpose(sp_collect_chans, [1,0,2]))

# pool by channels
pooling_chunks = images2neibs(next_in, th_pool_ch_shape)

# now each line of pooling chunks can be max-abs and argmax-abs
# we will need reshape to get actual pooling result from it
prep_pool = theano.function(
    [th_pool_in],
    pooling_chunks,
    givens=[
        (th_pool_sp_shape, s_pool_sp_shape),
        (th_pool_ch_shape, s_pool_ch_shape),
        (th_grouped_chs_shape, s_grouped_chs_shape)
    ]
)

# let's tr it
prepd = prep_pool(input)

print prepd.shape
