import jax
from jax import jit, vmap
import jax.numpy as jnp

from .utils import NODE, EDGE, GLOBAL

def _2Dshape(*args, N=None, DIM=None):
    return (arg.reshape(N, DIM) for arg in args)

class DYN():

    def __init__(self, gn, N=None, DIM=None):
        self.gn = gn
        self.N = N
        self.DIM = DIM

        @jit
        def PE(R, graph):
            nodes = graph.nodes
            _, V, ke, nodal_mass = nodes.position, nodes.velocity, nodes.ke, nodes.nodal_mass
            nodes = NODE(position=R, velocity=V, nodal_mass=nodal_mass, ke=ke)
            graph = graph._replace(nodes=nodes)
            return gn(graph).globals.PE[0]

        self.PE = PE

        @jit
        def KE(V, graph):
            nodes = graph.nodes
            R, _, ke, nodal_mass = nodes.position, nodes.velocity, nodes.ke, nodes.nodal_mass
            nodes = NODE(position=R, velocity=V, nodal_mass=nodal_mass, ke=ke)
            graph = graph._replace(nodes=nodes)
            return gn(graph).globals.KE[0]

        self.KE = KE

        @jit
        def KEPE(R, V, graph):
            nodes = graph.nodes
            _, _, ke, nodal_mass = nodes.position, nodes.velocity, nodes.ke, nodes.nodal_mass
            nodes = NODE(position=R, velocity=V, nodal_mass=nodal_mass, ke=ke)
            graph = graph._replace(nodes=nodes)
            globals = gn(graph).globals
            PE = globals.PE[0]
            KE = globals.KE[0]
            return KE, PE

        self.KEPE = KEPE

    def LAG(self, R, V, graph):
        KE, PE = self.KEPE(R, V, graph)
        return KE - PE

    def dL_dv(self, R, V, params):
        return jax.grad(self.LAG, 1)(*_2Dshape(R, V, N=self.N, DIM=self.DIM), params).flatten()

    def d2L_dv2(self, R, V, params):
        return jax.jacobian(self.dL_dv, 1)(R, V, params)

def acceleration(model, **kwargs):
    return accelerationFull(model.N, model.DIM, model.LAG, **kwargs)

def accelerationFull(n, Dim, lagrangian,
                     non_conservative_forces=None, 
                     external_force=None,
                     constraints=None, 
                     constant_mass_inv=None,
                     use_dissipative_forces=True
                    ):
    """ ̈q = M⁻¹(-C ̇q + Π + Υ - Aᵀ(AM⁻¹Aᵀ)⁻¹ ( AM⁻¹ (-C ̇q + Π + Υ + F ) + Adot ̇q ) + F )

    """
    def inv(x, *args, **kwargs):
        return jnp.linalg.pinv(x, *args, **kwargs)

    if non_conservative_forces is None:
        def non_conservative_forces(x, v, params): return 0.0
    if external_force is None:
        def external_force(x, v, t, params): return 0.0
    if constraints is None:
        def constraints(x, v, params): return jnp.zeros((1, n*Dim))
    if constant_mass_inv is None:
        # M⁻¹ = (∂²L/∂²v)⁻¹
        def getM_1(x, v, params): return inv(d2L_dv2(x, v, params))
    else:
        def getM_1(x, v, params): return jnp.array(constant_mass_inv)

    def _1Dshape(*args):
        return (arg.flatten() for arg in args)

    def _2Dshape(*args):
        return (arg.reshape(n, Dim) for arg in args)

    def _vec(*args):
        return (arg.reshape(-1, 1) for arg in args)

    def _shape(*args):
        return (arg.reshape(n*Dim, n*Dim) for arg in args)

    def dL_dv(R, V, params):
        return jax.grad(lagrangian, 1)(*_2Dshape(R, V), params).flatten()

    def d2L_dv2(R, V, params):
        return jax.jacobian(dL_dv, 1)(R, V, params)
    
    def fn(x, v, t, params):
        x_vec, v_vec = x.flatten(), v.flatten()
        V, = _vec(v)

        # if constant_mass_inv is None:
        #     # M⁻¹ = (∂²L/∂²v)⁻¹
        #     M = d2L_dv2(x_vec, v_vec, params)
        #     M_1 = inv(M)
        # else:
        #     M_1 = constant_mass_inv

        M_1 = getM_1(x_vec, v_vec, params)
        
        # Π = ∂L/∂x
        Π = jax.grad(lagrangian, 0)(x, v, params)
        Π, = _vec(Π)

        if use_dissipative_forces:
            # C = ∂²L/∂v∂x
            C = jax.jacobian(jax.jacobian(lagrangian, 1),
                             0)(x, v, params)
            C, = _shape(C)
            dis_f = -C @ V
        else:
            dis_f = 0.0

        Υ = non_conservative_forces(x, v, params)
        F = external_force(x, v, t, params)
        A = constraints(x_vec, v_vec, params)

        Aᵀ = A.T
        AM_1 = A @ M_1

        Ax = jax.jacobian(constraints, 0)(x_vec, v_vec, params)
        Adot = Ax @ v_vec

        FF = dis_f + Π + Υ + F
        xx = (AM_1 @ (FF) + Adot @ V)
        tmp = Aᵀ @ inv(AM_1 @ Aᵀ) @ xx
        out = M_1 @ (FF - tmp)

        return next(_2Dshape(out))

    return fn



