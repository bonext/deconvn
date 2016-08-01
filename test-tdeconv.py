import numpy as np

import theano
from theano.tensor.nnet.neighbours import images2neibs
import theano.tensor as T
floatX = theano.config.floatX

input = np.random.rand(2, 2, 4, 6).astype(theano.config.floatX)
in_shape = input.shape
neib_shape = [2, 3]

i2n_in = T.tensor4()
t_get_spatial_ns = theano.function(
    [i2n_in],
    images2neibs(i2n_in, neib_shape)
)

output_spatial = t_get_spatial_ns(input)
print output_spatial.shape

#print input
#print "---"
#for i in range(output_spatial.shape[0]):
#    print output_spatial[i, :]


# output_spatial lines per channel
lpc = np.prod(input.shape[-2:])/np.prod(neib_shape)

next_in = output_spatial.reshape(output_spatial.shape[0]/lpc, lpc, output_spatial.shape[-1])
next_in = next_in.transpose(1,0,2)

c_shape = [2, next_in.shape[-1]]
t_get_ch_ns = theano.function(
    [i2n_in],
    images2neibs(i2n_in, c_shape)
)

out_ch = t_get_ch_ns(next_in[np.newaxis,:,:,:])

print input
print "---"
#print next_in
for i in range(out_ch.shape[0]):
    print out_ch[i]
