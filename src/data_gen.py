import jax.numpy as jnp
import numpy as np
from .dynamics import acceleration
from .utils import solve_dynamics, get_apply, NODE, EDGE, GLOBAL

def get_func(F, f_args, ftype=None):
    if ftype is None:
        rate = f_args[0]
        def external_force(x, v, t, graph):
            _F = F * jnp.exp(-rate * t)
            return _F.reshape(-1, 1)
        return external_force
    elif ftype == "sin":
        w = f_args[0]
        def external_force(x, v, t, graph):
            _F = F * jnp.sin(w * t)
            return _F.reshape(-1, 1)
        return external_force

def make_F(shape, x=[], y=[], z=[]):
    F = np.zeros(shape)
    for i in x:
        F[i, 0] = 1.0
    for i in y:
        F[i, 1] = 1.0
    for i in z:
        F[i, 2] = 1.0
    return F.reshape(-1, 1)

def getF(loads, F0, shape):
    Fs = [make_F(shape, **load) for load in loads]
    F = np.hstack(Fs)
    return F0*F

def get_gen_traj(model, state, **acc_kwargs):
    def generate_traj(F, runs=100, stride=10, f_args=(1.0,), Ffunc=None):
        if ~isinstance(Ffunc, type(lambda x: x)):
            Ffunc = get_func(F, f_args, ftype=Ffunc)
        acc = acceleration(model,
                               external_force=Ffunc, **acc_kwargs)
        apply = get_apply(acc)
        traj = solve_dynamics(state, apply, runs=runs, stride=stride)
        return traj
    return generate_traj

