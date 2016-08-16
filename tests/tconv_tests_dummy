import numpy as np

# gotta check my theano impl of convolution and bwd convolution
filters = np.array(
    [[[[1,0,0],
       [0,0,0],
       [0,0,0]]],
     [[[0,0,0],
       [0,0,0],
       [0,0,-1]]]
    ],
    dtype=np.float32
)

print filters.shape
print filters

# test signal 1 (forward)
s1 = np.array(
    [[[[0,0,0,0,0],
       [0,0,1,0,0],
       [0,1,1,1,0],
       [0,0,1,0,0],
       [0,0,0,0,0]
    ]]],
    dtype=np.float32
)

print s1.shape
print s1

# test signal 2 (backward)
s2 = np.array(
    [[[[0,1,0],
       [0,1,0],
       [0,1,0]],
      [[0,0,0],
       [1,1,1],
       [0,0,0]]]],
    dtype=np.float32
)
print s2.shape
print s2

from tdeconv import Convolution
op = Convolution(filters.shape)
op.filters.set_value(filters)

print "Testing forward conv"
print op(s1)

print "Testing backward conv"
print op.T(s2)

"""
Convert this to test,
op(s1) result should be
[[[[1,1,0],
   [1,0,0],
   [0,0,0]],
  [[0,0,0],
   [0,0,-1],
   [0,-1,-1]]
]]

op.T(s2) result should be
[[[[0,0,0,0,1],
   [-1,-1,-1,0,0],
   [0,0,0,1,0],
   [0,0,0,1,0],
   [0,0,0,1,0]]]]
"""
