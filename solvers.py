import numpy as np

def prox_abs(z, rho):
    ret = np.zeros(z.shape)
    ret[z > rho] = z[z > rho] - rho
    ret[z < -rho] = z[z < -rho] + rho
    return ret


def pxg_step(z, prox, grad, rho):
    return prox(z - rho * grad(z), rho)


def prox_grad(prox, grad, errf, z0, rho0, rho_rate, rho_min, n_iters, verbose=False):
    z = z0.copy()
    errs = []
    rho_curr = rho0
    for it in range(n_iters):
        if rho_curr < rho_min:
            break
        znew = pxg_step(z, prox, grad, rho_curr)
        err = errf(znew)
        if len(errs) == 0:
            errs.append(err)
        else:
            while err >= errs[-1]:
                # reduce rho
                rho_curr *= rho_rate
                if rho_curr < 1e-6:
                    if verbose:
                        print 'rho way too small, exiting'
                    break
                if verbose:
                    print 'new rho: ', rho_curr
                znew = pxg_step(z, prox, grad, rho_curr)
                err = errf(znew)
            z = znew
            errs.append(err)
            if verbose:
                print 'it: ', it, ' err: ', err
    return z, errs
