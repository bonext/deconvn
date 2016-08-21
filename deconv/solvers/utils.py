import theano
import theano.tensor as T

import numpy as np

from .. import tdeconv_utils

if __name__ == "__main__":
    floatX = theano.config.floatX

    t_z = T.tensor4()
    t_z0 = T.tensor4()
    t_f = T.tensor4()
    t_y = T.tensor4()
    t_lm = T.scalar()
    t_rho = T.scalar()
    t_ista_iters = T.iscalar()

    y = np.zeros((1, 3, 16, 16), dtype=floatX)
    y[:, 0, :, :] = 1
    y[:, 1, y.shape[2] / 3:2 * y.shape[2] / 3, y.shape[3] / 3:2 * y.shape[3] / 3] = 1
    y[:, 2, y.shape[2] / 3:2 * y.shape[2] / 3, y.shape[3] / 3:2 * y.shape[3] / 3] = 1
    filters = np.random.randn(3, 2, 5, 5).astype(floatX)

    ptp = theano.function(
        [t_z, t_f],
        tdeconv_utils.t_mk_conv_transpose(
            tdeconv_utils.t_mk_conv(t_z, t_f),
            t_f
        )
    )

    z0 = np.random.randn(
        y.shape[0],
        filters.shape[1],
        y.shape[2]+filters.shape[2]-1,
        y.shape[3]+filters.shape[3]-1
    ).astype(floatX)
    z = z0.copy()

    curr=z
    norm_prev=0
    for ix in range(100):
        next = ptp(curr, filters)
        next = next / np.sqrt(np.sum(next**2))
        curr = next
        norm_curr = np.sqrt(np.sum((next-ptp(next, filters))**2))
        norm_prev = norm_curr
    print norm_prev