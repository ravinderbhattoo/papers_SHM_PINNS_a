from .units import U, ustrip, convert
from jax import numpy as jnp


# Setting units and CONSTANTS
U_Acceleration = U.meter / U.second / U.second
U_Velocity = U.meter / U.second
U_Force = U.kg * U_Acceleration
U_Area = U.meter ** 2
U_Volume = U.meter ** 3
U_Density = U.kg / U_Volume
U_Length = U.meter
U_Time = U.second


def getExample(name):
    return DICT[name]()

def example1():
    PREFERED_UNITS = [U.cm, U.g, U.ms]
    from .truss_types import BoxTruss

    E0 = 7*1.0e9 * U_Force / U_Area
    ρ = 2770 * U_Density
    A0 = 0.01 * U_Area
    F0 = 1.0e6 * U_Force
    width = 1.52 * U_Length
    height = 1.52 * U_Length
    rate = 1.0e3 / U_Time

    items = (E0, ρ, A0, F0, width, height, rate)
    p_units = [U.cm, U.g, U.ms]

    items = ustrip(*convert(p_units, *items))

    E0, ρ, A0, F0, width, height, rate = items

    EA0 = E0*A0

    truss = BoxTruss(n = 6, width=width, height=height)
    
    N, DIM = truss.nnodes, truss.dim
    
    ##############################################################################
    # Define contraints (supports)
    ##############################################################################

    def constraints(x, v, graph):
        """Produces A(q) i.e. differentiating holonomic constraints wrt time
        and converting it to form of A(q)q̇ = 0.
        For e.g. in a case of simple pendulum x2+y2=l2
        x.ẋ + y.ẏ = 0
        [x y][ẋ ẏ].T = 0
        A(q) = [x y]
        """
        A_q =  jnp.array([
                            [x[0]-0.0, 0.0] + [0.0]*26,
                            [0.0, x[1]-0.0] + [0.0]*26,
                            [0.0]*12 + [x[12] - 6*width, 0.0] + [0.0]*14,
                            [0.0]*12 + [0.0, x[13]-0.0] + [0.0]*14,
                        ]).reshape(4, -1)
        return A_q.reshape(-1, N*DIM)

    return truss, (E0, ρ, A0, F0, rate), EA0, constraints


def example2():
    PREFERED_UNITS = [U.cm, U.g, U.ms]
    from .truss_types import Bar25

    E0 = 210*1.0e9 * U_Force / U_Area
    ρ = 7830 * U_Density
    A0 = 0.0025 * U_Area
    F0 = 1.0e6 * U_Force
    rate = 1.e3 / U.second

    a1 = 8.0 * U_Length
    a2 = 3.0 * U_Length
    h1 = h2 = 4.0*U_Length

    bwidth = 8 * U_Length
    items = (E0, ρ, A0, F0, rate, a1, a2, h1, h2, bwidth)

    items = ustrip(*convert(PREFERED_UNITS, *items))

    # Converted constants
    E0, ρ, A0, F0, rate, a1, a2, h1, h2, bwidth = items
    EA0 = E0*A0

    truss = Bar25(a1=a1, a2=a2, h1=h1, h2=h2)
    
    N, DIM = truss.nnodes, truss.dim
    
    ##############################################################################
    # Define contraints (supports)
    ##############################################################################

    def constraints(x, v, graph):
        """Produces A(q) i.e. differentiating holonomic constraints wrt time
        and converting it to form of A(q)q̇ = 0.
        For e.g. in a case of simple pendulum x2+y2=l2
        x.ẋ + y.ẏ = 0
        [x y][ẋ ẏ].T = 0
        A(q) = [x y]
        """
        d = bwidth / 2 
        A_q = jnp.array([
                            [x[0]-(-d), 0.0, 0.0] + [0.0]*27,
                            [0.0, x[1]-(-d), 0.0] + [0.0]*27,
                            [0.0, 0.0, x[2]-0.0] + [0.0]*27,

                [0.0]*3 + [x[3]-(-d), 0.0, 0.0] + [0.0]*24,
                [0.0]*3 + [0.0, x[4]-(d), 0.0] + [0.0]*24,
                [0.0]*3 + [0.0, 0.0, x[5]-0.0] + [0.0]*24,

                [0.0]*6 + [x[6]-(d), 0.0, 0.0] + [0.0]*21,
                [0.0]*6 + [0.0, x[7]-(-d), 0.0] + [0.0]*21,
                [0.0]*6 + [0.0, 0.0, x[8]-0.0] + [0.0]*21,

                [0.0]*9 + [x[9]-(d), 0.0, 0.0] + [0.0]*18,
                [0.0]*9 + [0.0, x[10]-(d), 0.0] + [0.0]*18,
                [0.0]*9 + [0.0, 0.0, x[11]-0.0] + [0.0]*18,

        ])
        return A_q.reshape(-1, N*DIM)

    return truss, (E0, ρ, A0, F0, rate), EA0, constraints


DICT = dict(
    EXAMPLE1 = example1,
    EXAMPLE2 = example2,
    )
    