import jax.numpy as jnp
import numpy as np
from jax import nn, jit, vmap, grad
from . import utils
import matplotlib.pyplot as plt

def get_nodes_edges(truss, E0, A0, ρ, damage=None, offset=None):
    node_position = jnp.array(truss.nodes["position"])
    N = len(node_position)
    edge_index = truss.edges["index"]
    senders = jnp.array(list(edge_index[:, 0]))
    receivers = jnp.array(list(edge_index[:, 1]))
    n_node = jnp.array([N])

    n_edge = jnp.array([len(senders)])
    L = jnp.sqrt(((node_position[receivers] - node_position[senders])**2).sum(axis=1))

    e_type = jnp.array(np.arange(n_edge.item())).reshape(-1, )
    e_type = nn.one_hot(e_type, num_classes=len(senders))

    if damage is None:
        damage = 0.0*A0

    if offset is None:
        offset = 0.0*A0


    EA = np.array([E0*A0]*len(senders))
    dd = damage/100
    EAd = EA * (1 - dd)
    A = L*0 + A0

    edges = utils.EDGE(EA = jnp.array(EAd), L=L, dx=L*0, pe=L*0, A=A, ρ=ρ, mass=ρ*A*L, type=e_type, offset=offset)

    nodes = utils.NODE(position=node_position, velocity=node_position*0,
                                     nodal_mass = jnp.array([0.0]* len(node_position)),
                                     ke = jnp.array([0.0]*len(node_position)))
    full_dict = dict(nodes=nodes,
                  edges=edges,
                  senders=senders,
                  receivers=receivers,
                  n_node=n_node,
                  n_edge=n_edge)

    return nodes, edges, full_dict


def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
    s_X = sent_attributes.position
    r_X = received_attributes.position
    l = jnp.sqrt((jnp.square(s_X - r_X)).sum(axis=1))
    EA, L, A, ρ, offset = edges.EA, edges.L, edges.A, edges.ρ, edges.offset

    dx = jnp.abs(l - L)
    dx = jnp.where(jnp.abs(dx/L) < offset, 0.0, dx)
    pe = 0.5 * EA / L * jnp.square(dx)

    edges = utils.EDGE(EA = EA, L=L, dx=dx, pe=pe, A=A, ρ=ρ, mass=A*L*ρ,
                       type=edges.type, offset=offset)
    return edges

def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
    nodal_mass = received_attributes.mass / 2 + sent_attributes.mass / 2
    V = nodes.velocity
    ke = 0.5 * nodal_mass * jnp.square(V).sum(axis=1)
    nodes = utils.NODE(nodal_mass=nodal_mass, position=nodes.position,
                       velocity=nodes.velocity, ke=ke)
    return nodes

def update_global_fn(node_attributes, edge_attribtutes, globals_):
    globals_ = utils.GLOBAL(PE=edge_attribtutes.pe, KE=node_attributes.ke)
    return globals_


def plot_traj_energy(trajs, tmodel, graph_truss):
    for traj in trajs:
        _t = traj[-2]
        _kepe = [tmodel.KEPE(traj[0][i], traj[1][i], graph_truss) for i in range(len(_t))]
        _ke = jnp.array([i[0] for i in _kepe])
        _pe = jnp.array([i[1] for i in _kepe])
        _t = traj[-2]

        fig, ax = plt.subplots()

        plt.plot(_t, _ke, label="KE")
        plt.plot(_t, _pe, label="PE")
        plt.plot(_t, _ke + _pe, label="Ham")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc=(-0.1, 1.02), ncols=4)
        plt.show()






















#