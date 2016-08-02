import numpy as np

import theano
from theano.tensor.nnet.neighbours import images2neibs
import theano.tensor as T
floatX = theano.config.floatX

input = np.random.rand(2, 2, 4, 6).astype(theano.config.floatX)
in_shape = input.shape
neib_shape = [2, 3]

# lets do it all in one step
pool_shape = [2, 2, 3] # channels x h x w

pool_in = T.tensor4() # pool input batch x channels x H x W

# collect spatial neighbourhoods
sp_pool = images2neibs(pool_in, pool_shape[-2:])

# spatial lines per channel - can be prepared in advance
lpc = np.prod(input.shape[-2:])/np.prod(pool_shape[-2:])

# collect channels in a separate dimension
sp_pool_shape = () # (sp_pool.shape[0]/lpc, lpc, sp_pool.shape[-1])
sp_collect_chans = sp_pool.reshape(sp_pool_shape)
next_in = T.shape_padleft(sp_collect_chans.transpose(1,0,2))

# pool by channels
pooling_chunks = images2neibs(next_in, [pool_shape[0], next_in.shape[-1]])

# now each line of pooling chunks can be max-abs and argmax-abs
# we will need reshape to get actual pooling result from it


