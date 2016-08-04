import numpy as np
import tdeconv, deconv

import theano
floatX = theano.config.floatX

pooling_shape = (2, 2, 3) # channels, height, width
P = tdeconv.Pooling(pooling_shape)

pool_input = np.random.randn(1, 4, 4, 6).astype(floatX)

# reference numpy pooling op
P_ref = deconv.Pooling(pool_input.shape[1:], pooling_shape)

print "Testing theano pooling"
print "input"
print pool_input

print "reference result"
print P_ref.P(pool_input[0])

print "result"
res = P.P(pool_input)
print res
