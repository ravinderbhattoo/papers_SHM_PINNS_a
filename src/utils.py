import jax
from jax import jit, vmap
import jax.numpy as jnp

import numpy as np
from collections import namedtuple

EDGE = namedtuple('EDGE', 'EA,L,dx,pe,A,ρ,mass,type,offset')
NODE = namedtuple('NODE', 'position,velocity,nodal_mass,ke')
GLOBAL = namedtuple('GLOBAL', 'PE,KE')

def dec2bin(x, length=-1):
    if isinstance(x, int):
        x = jnp.array([x])
    n = len(x)
    y = jnp.unpackbits(x.view('uint8'), bitorder='little').reshape(n, -1)[:, :length]
    return jnp.array(y, dtype=float)



def index_traj(traj, index, shift=0.0):
    n = len(traj[0])
    a, b, c = index
    if b==-1:
        b = n
    l1 = list(range(a, b, c)) + [-1]
    index = np.array(l1).astype(int)
    items = []
    for item in traj[:3]:
        items += [jnp.array([item[i] for i in index])]
    items[0] = items[0] + shift
    return (*items, traj[3], jnp.array([traj[4][i] for i in index]), traj[5])


def newmark_beta(state, acc_fn, γ=0.5, β=0.25):
    Rn, Vn, An, graph, tn, dt = state
    An1 = acc_fn(Rn, Vn, tn+dt, graph)
    Vn1 = Vn + (1-γ)*dt*An + γ*dt*An1
    Rn1 = Rn + dt*Vn + 0.5*dt*dt*((1-2*β)*An+2*β*An1)
    return (Rn1, Vn1, An1, graph, tn+dt, dt)

def velocity_verlet(state, acc_fn):
    Rn, Vn, An, graph, tn, dt = state
    Rn1 = Rn + Vn * dt + 0.5 * An * dt * dt
    An1 = acc_fn(Rn1, Vn, tn+dt, graph)
    Vn1 = Vn + 0.5 * (An + An1) * dt
    return (Rn1, Vn1, An1, graph, tn+dt, dt)

def get_apply(acc_fn, integrator=velocity_verlet):
    def apply(state):
        return integrator(state, acc_fn)
    return apply


def solve_dynamics(init_state, apply, runs=100, stride=10):
    step = jit(lambda i, state: apply(state))

    def f(state):
        y = jax.lax.fori_loop(0, stride, step, state)
        return y, y

    def func(state, i): return f(state)

    @jit
    def scan(init_state):
        return jax.lax.scan(func, init_state, jnp.array(range(runs)))

    final_state, traj = scan(init_state)
    return traj


def add_noise(k, _ACCELERATION, key = jax.random.PRNGKey(0)):

    _RUNS = _ACCELERATION.shape[-2]

    def randn(x):
        return x*jax.random.normal(key, (_RUNS, ))

    def fn1(x):
        return vmap(randn)(x).T

    σ = k * _ACCELERATION.std(axis=(1, ))
    noise = vmap(fn1)(σ)
    ACCELERATION = _ACCELERATION + noise
    return ACCELERATION

def MLP(x, layers):
    for layer in layers:
        x = layer(x)
    return x

def getDamage(model, graph):
    myMLP = lambda x: MLP(x, model.layers[0])
    damage = vmap(myMLP)(graph.edges.type).reshape(-1, )
    return damage

def getOffset(model, graph):
    myMLP = lambda x: MLP(x, model.layers[1])
    offset = vmap(myMLP)(graph.edges.type).reshape(-1, )
    return offset
    
def getEA(model, graph, EA_fn):
    k = getDamage(model, graph)
    damaged_stiff = EA_fn(k.reshape(-1, ))
    return damaged_stiff
