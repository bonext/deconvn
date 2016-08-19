import theano
import theano.tensor as T
from .. import tdeconv_utils

def ista_iteration(z, f, y, lm, rho):
    # TODO: this has to be a reconstruction operator from switches and filters
    r = tdeconv_utils.t_mk_conv(z, f) - y
    # TODO: this has to be a synthesis operator from switches and filters
    dz = lm * tdeconv_utils.t_mk_conv_transpose(r, f)
    z_hat = z - rho * dz
    return T.maximum(abs(z_hat) - rho, 0)*T.sgn(z_hat)

if __name__ == "__main__":
    # this shows how to do ISTA iterations with scan
    # we need however a good guess for rho
    # it can be determined from operator, but that is probably
    t_z = T.tensor4()
    t_z0 = T.tensor4()
    t_f = T.tensor4()
    t_y = T.tensor4()
    t_lm = T.scalar()
    t_rho = T.scalar()
    t_ista_iters = T.iscalar()

    values, updates = theano.scan(
        fn=ista_iteration,
        outputs_info=t_z0,
        non_sequences=[t_f, t_y, t_lm, t_rho],
        n_steps=t_ista_iters
    )

    do_ista = theano.function(
        [t_z0, t_f, t_y, t_lm, t_rho, t_ista_iters],
        values[-1]
    )