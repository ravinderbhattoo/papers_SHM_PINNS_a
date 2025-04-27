import numpy as np
import jax.numpy as jnp
from ..data_gen import get_func

def get_constraints(width):
    def constraints(x, v, graph):
        """Produces A(q) i.e. differentiating holonomic constraints wrt time
        and converting it to form of A(q)q̇ = 0.
        For e.g. in a case of simple pendulum x2+y2=l2
        x.ẋ + y.ẏ = 0
        [x y][ẋ ẏ].T = 0
        A(q) = [x y]
        """
        C_matrix = jnp.zeros((4, 28))
        C_matrix = C_matrix.at[jnp.array([0, 1, 2, 3]), jnp.array([0, 1, 12, 13])].set(jnp.array([x[0], x[1], x[12] - 6*width, x[13]]))
        return C_matrix
    return constraints


def get_eF(ind, F0, shape, func=None, f_args=None):

    F = np.zeros(shape)
    F[ind, 1] = 1.0
    F = F0 * F.reshape(-1, 1)
    if func is None:
        return get_func(F, f_args, ftype=None)
    elif isinstance(func, str):
        return get_func(F, f_args, ftype=func)

    else:
        def fn(x, v, t, graph):
            return F * func(t)
        return fn
