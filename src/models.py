import jax
import warnings

from jax import jit, vmap
import jax.numpy as jnp
from jraph import GraphNetwork, GraphsTuple

from typing import Any, Callable, Iterable, Mapping, Optional, Union, Sequence
from jaxtyping import Array, Float, Int, PyTree, Bool
from functools import partial

import jax.tree_util as tree
import equinox as eqx

from .dynamics import accelerationFull

from .utils import solve_dynamics, get_apply, NODE, EDGE, GLOBAL, MLP
from .data_gen import get_func, make_F, getF

def squareplus(x):
    return jax.lax.mul(0.5, jax.lax.add(x, jax.lax.sqrt(jax.lax.add(jax.lax.square(x), 4.0))))


class TrussGraphModel(eqx.Module):
    layers: list
    N: Int
    Ne: Int
    dim: Int
    runs: Int
    stride: Int
    NODAL_MASS0: Array
    M_1: Array
    external_force: Callable
    constraints: Callable
    EA_fn: Callable
    activation: Callable
    activation2: Callable
    hidden_dim: Int
    latent_dim: Int
    use_dissipative_forces: Bool

    def __init__(self, key,
                 N=None, Ne=None, dim=2,
                 external_force=None,
                 use_dissipative_forces=True,
                 EA_fn=None,
                 constraints=None,
                 runs=10, stride=10,
                 NODAL_MASS0=None, M_1=None,
                 activation2=jax.nn.leaky_relu,
                 activation=jax.nn.leaky_relu,
                 hidden_dim=8,
                 latent_dim=8,
                 override=True,
                ):
        key2, key1 = jax.random.split(key, 2)
        self.runs, self.stride, self.N, self.dim, self.Ne = runs, stride, N, dim, Ne
        self.external_force = external_force
        self.constraints = constraints
        self.use_dissipative_forces = use_dissipative_forces
        self.EA_fn = EA_fn
        self.NODAL_MASS0 = NODAL_MASS0
        self.activation = activation
        self.activation2 = activation2
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.M_1 = M_1

        if override:
            warnings.warn('Setting M_1 to None.')
            self.M_1 = None

        damage_layer = [
            eqx.nn.Linear(Ne, self.hidden_dim, key=key2),
            squareplus,
            eqx.nn.Linear(self.hidden_dim, 1, key=key2),
            lambda x: x - 10.0,
            jax.nn.sigmoid,
            ]

        z_layer = [
            eqx.nn.Linear(1, self.hidden_dim, key=key2),
            jax.nn.leaky_relu,
            eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=key2),
            jax.nn.leaky_relu,
            eqx.nn.Linear(self.hidden_dim, 1, key=key2),
            ]

        self.layers = [damage_layer, z_layer]

    def __call__(self, _state, train=True):

        MLP_damage = lambda x: MLP(x, self.layers[0])
        MLP_z = lambda x: MLP(x, self.layers[1])

        if train:
            def getEA(EA, etype, dx):
                print("recompile getEA")
                damage = jax.vmap(MLP_damage)(etype).reshape(-1, )
                EA = self.EA_fn(damage).reshape(-1, )

                # embd = jax.vmap(self.layers[1][0])(etype) / 1000
                # xz = jnp.hstack((embd, dx.reshape(-1, 1)))

                dx2 = jnp.square(dx)
                # z = jax.vmap(MLP_z)(dx.reshape(-1, 1)).reshape(-1, )

                z = dx2
                return EA, z
        else:
            def getEA(EA, _, dx):
                z = dx ** 2.0
                return EA, z


        def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
            s_X = sent_attributes.position
            r_X = received_attributes.position
            l = jnp.sqrt((jnp.square(s_X - r_X)).sum(axis=1))
            EAd, L, offset = edges.EA, edges.L, edges.offset

            # EA, offset = getEA(EAd, edges.type, offset)
            # dx = l - L
            # dx = jnp.where(jnp.abs(dx/L) < offset, 0.0, dx)
            # z = EA * jnp.square(dx)
            # pe = 0.5 * 1 / L * z

            dx = jnp.abs(l - L)
            EA, z = getEA(EAd, edges.type, dx)
            pe = 0.5 * 1 / L * EA * z
            edges = EDGE(EA = EAd, L=L, dx=dx, pe=pe, A=edges.A,
                         ρ=edges.ρ, mass=edges.ρ*edges.A*L, type=edges.type, offset=offset)
            return edges

        def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
            nodal_mass = received_attributes.mass / 2 + sent_attributes.mass / 2
            nodal_mass += self.NODAL_MASS0
            V = nodes.velocity
            ke = 0.5 * nodal_mass * jnp.square(V).sum(axis=1)
            nodes = NODE(nodal_mass=nodal_mass, position=nodes.position, velocity=nodes.velocity, ke=ke)
            return nodes

        def update_global_fn(node_attributes, edge_attribtutes, globals_):
            globals_ = GLOBAL(PE= edge_attribtutes.pe, KE=node_attributes.ke)
            return globals_

        gn = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)

        def KEPE(R, V, graph):
            print("recompile KEPE")
            nodes = graph.nodes
            _, _, ke, nodal_mass = nodes.position, nodes.velocity, nodes.ke, nodes.nodal_mass
            nodes = NODE(position=R, velocity=V, nodal_mass=nodal_mass, ke=ke)
            graph = graph._replace(nodes=nodes)
            globals = gn(graph).globals
            PE = globals.PE[0]
            KE = globals.KE[0]
            return KE, PE

        def Lag(R, V, graph):
            KE, PE = KEPE(R, V, graph)
            return KE - PE

        state, F, (f_args, ftype) = _state
        acc = accelerationFull(self.N, self.dim, Lag, external_force=get_func(F, f_args, ftype=ftype),
                               constraints=self.constraints, use_dissipative_forces=self.use_dissipative_forces,
                               constant_mass_inv=self.M_1)
        apply = get_apply(acc)
        def fn(state):
            print("recompile forward")
            return solve_dynamics(state, apply, runs=self.runs, stride=self.stride)
        traj = fn(state)
        return traj

def copy_model(model, **kwargs):
    key = jax.random.PRNGKey(0)
    ds = dict(N=model.N, Ne=model.Ne, dim=model.dim, external_force=model.external_force,
              constraints=model.constraints, EA_fn=model.EA_fn,
              runs=model.runs, stride=model.stride,
              NODAL_MASS0=model.NODAL_MASS0,
              M_1=model.M_1,
              activation=model.activation,
              activation2=model.activation2,
              hidden_dim=model.hidden_dim,
              )

    ds.update(kwargs)
    new_model = TrussGraphModel(key, **ds)
    new_model.layers[:] = model.layers[:]
    return new_model
