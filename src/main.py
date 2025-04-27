##############################################################################
# import packages
##############################################################################

from . import tt, pt, utils, dyn, data_gen, models, train, units, io, plot_tools, examples
        
plt = pt.plt
plt.rcParams["figure.dpi"] = 70
plt.rcParams["figure.figsize"] = [6.0, 6.0]

import jax.numpy as jnp
import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
from jax import nn, jit, vmap, grad
from jraph import GraphsTuple, GraphNetwork
import os
from shadow import panel

RUNS = 0

def main(NAME, DSEED, ndamage, fdamage, Fnodes, SSEED, nsensors):

    ##############################################################################
    # Setting constants
    ##############################################################################

    global RUNS 
    RUNS += 1
    print(RUNS)

    subdir = f"{DSEED}_{SSEED}_{ndamage}_{nsensors}"
    f_obj = io.FileIOSTR(NAME, subdir)

    def savemyfig(name, fig=None):
        filename = f_obj.fig(name+".png")
        plt.savefig(filename, dpi=300)

    def counter():
        return iter(range(1, 100))

    # Setting units and CONSTANTS
    U = units.U
    U_Acceleration = U.meter / U.second / U.second
    U_Velocity = U.meter / U.second
    U_Force = U.kg * U_Acceleration
    U_Area = U.meter ** 2
    U_Volume = U.meter ** 3
    U_Density = U.kg / U_Volume
    U_Length = U.meter
    U_Time = U.second

    PREFERED_UNITS = [U.cm, U.g, U.ms]

    ##############################################################################
    # Constants
    ##############################################################################

    # E0 = 210*1.0e9 * U_Force / U_Area
    # ρ = 7830 * U_Density
    # A0 = 0.0025 * U_Area
    # F0 = 1.0e6 * U_Force
    # rate = 1.e3 / U.second

    # a1 = 8.0 * U_Length
    # a2 = 3.0 * U_Length
    # h1 = h2 = 4.0*U_Length
    # truss = tt.Bar25(a1=a1, a2=a2, h1=h1, h2=h2)

    # bwidth = 8 * U_Length
    # items = (E0, ρ, A0, F0, rate, a1, a2, h1, h2, bwidth)

    # items = units.ustrip(*units.convert(PREFERED_UNITS, *items))

    # # Converted constants
    # E0, ρ, A0, F0, rate, a1, a2, h1, h2, bwidth = items
    # EA0 = E0*A0

    truss, items, EA0, constraints = examples.getExample(NAME)
    E0, ρ, A0, F0, rate = items

    ##############################################################################
    # Loading truss item
    ##############################################################################

    truss.plot(
            text_kwargs={"fontsize": 4}, 
    )
    savemyfig("truss_image")

    if truss.dim == 3:
        truss.plot(ThreeD=True,
                text_kwargs={"fontsize": 4}, 
                )
        savemyfig("truss_image_3d")


    ##############################################################################
    # Make a graph obj from truss
    ##############################################################################

    node_position = jnp.array(truss.nodes["position"])
    N = len(node_position)
    edge_index = truss.edges["index"]
    senders = jnp.array(list(edge_index[:, 0]))
    receivers = jnp.array(list(edge_index[:, 1]))
    n_node = jnp.array([N])

    n_edge = jnp.array([len(senders)])
    L = jnp.sqrt(((node_position[receivers] - node_position[senders])**2).sum(axis=1))

    e_type = jnp.array(np.arange(len(senders))).reshape(-1, )
    e_type = nn.one_hot(e_type, num_classes=len(senders))
    # e_type = dec2bin(jnp.arange(len(senders)), length=5)

    np.random.seed(DSEED)
    damage = 40.0*np.random.rand(len(senders))
    damage -= damage.min()


    _ch = np.random.choice(len(damage), ndamage, replace=False)
    damage[_ch] = 0.0

    _ch = np.random.choice(len(damage), fdamage, replace=False)
    damage[_ch] = 100.0
    # damage[:] = 0.0

    EA = np.array([E0*A0]*len(senders))
    dd = damage/100
    EAd = EA * (1 - dd)

    A = L*0 + A0

    offset = np.abs(np.random.rand(len(senders)) / 1000)
    _ch = np.random.choice(len(damage), len(senders) - int(1), replace=False)
    offset[:] = 0.0

    edges = utils.EDGE(EA = jnp.array(EAd), L=L, dx=L*0, pe=L*0, A=A, ρ=ρ, mass=ρ*A*L, type=e_type, offset=offset)

    nodes = utils.NODE(position=node_position, velocity=node_position*0,
                                    nodal_mass = jnp.array([0.0]* len(node_position)), 
                                    ke = jnp.array([0.0]*len(node_position)))
    globals = utils.GLOBAL(PE = jnp.array([0.0]),
                    KE =  jnp.array([0.0]))


    key = jax.random.PRNGKey(0)
    graph_truss = GraphsTuple(nodes=nodes,
                    edges=edges,
                    senders=senders,
                    receivers=receivers,
                    n_node=n_node,
                    n_edge=n_edge,
                    globals=globals)


    ##############################################################################
    # define Lagrangian for the truss system
    ##############################################################################

    def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
        s_X = sent_attributes.position
        r_X = received_attributes.position
        l = jnp.sqrt((jnp.square(s_X - r_X)).sum(axis=1))   
        EA, L, A, ρ, offset = edges.EA, edges.L, edges.A, edges.ρ, edges.offset
    
        dx = l - L
        dx = jnp.where(jnp.abs(dx/L) < offset, 0.0, dx)
        pe = 0.5 * EA / L * jnp.square(dx)
        
        edges = utils.EDGE(EA = EA, L=L, dx=dx, pe=pe, A=A, ρ=ρ, mass=A*L*ρ, type=edges.type, offset=offset)
        return edges
        
    def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
        nodal_mass = received_attributes.mass / 2 + sent_attributes.mass / 2
        V = nodes.velocity
        ke = 0.5 * nodal_mass * jnp.square(V).sum(axis=1)
        nodes = utils.NODE(nodal_mass=nodal_mass, position=nodes.position, velocity=nodes.velocity, ke=ke)
        return nodes

    def update_global_fn(node_attributes, edge_attribtutes, globals_):
        globals_ = utils.GLOBAL(PE=edge_attribtutes.pe, KE=node_attributes.ke) 
        return globals_

    gn = GraphNetwork(update_edge_fn, update_node_fn, update_global_fn)
    graph_truss = gn(graph_truss)

    N, DIM = nodes.position.shape
    tmodel = dyn.DYN(gn, N=N, DIM=DIM)


    ##############################################################################
    # calculating pe and ke
    ##############################################################################

    R = node_position
    V = node_position*0+1.0

    print("PE", tmodel.PE(R*1.1, graph_truss))
    print("KE", tmodel.KE(V, graph_truss))
    print("Lag", tmodel.LAG(R, V, graph_truss))

    # Reset to 0.0
    V = V*0.0

        
    ##############################################################################
    # Define force function (excitation force)
    ##############################################################################

    f_nodes = Fnodes

    def get_eF(ind):
        F = np.zeros(nodes.position.shape)
        F[ind, 1] = 1.0
        F = -F0*F.reshape(-1, 1)

        def external_force(x, v, t, graph):
            return F * jnp.exp((-rate)* t)
        
        return external_force

    external_force = get_eF(jnp.array(f_nodes))

    ##############################################################################
    # Calculating MASS matrix
    ##############################################################################
    x_vec, v_vec = R.flatten(), V.flatten()

    M = tmodel.d2L_dv2(x_vec, v_vec, graph_truss)
    M_1 =  jnp.linalg.pinv(M)


    ##############################################################################
    # Define acceleration function using Euler-Lagrange equation
    ##############################################################################
    acc = dyn.acceleration(tmodel,
                        external_force=external_force, 
                        constraints=constraints, 
                        use_dissipative_forces=False, 
                        constant_mass_inv=M_1
                        )

    print(F0 / graph_truss.nodes.nodal_mass, acc(R, 0*R, 0, graph_truss)[:, 1])


    ##############################################################################
    # Define simulation paramters and do example dynamics
    ##############################################################################
    RUNS = 100
    STRIDE = 10
    DT = 1.0

    dt = DT/STRIDE
    print(f"dt: {dt} ms")

    t = 0.0

    A = acc(R, 0*V, t, graph_truss)
    state = (R, 0*V, A, graph_truss, t, dt)

    apply = utils.get_apply(acc)
    traj = utils.solve_dynamics(state, apply, runs=RUNS, stride=STRIDE)

    print(f"Total time: {dt*RUNS*STRIDE} ms")


    ##############################################################################
    # Plot simulation result
    ##############################################################################
    ff = vmap(lambda x: external_force(None, None, x, None))(traj[-2])

    fig, ax = panel(1, 1, figsize=(6, 6), dpi=100)
    plt.plot(ff.reshape(RUNS, -1, DIM)[:, f_nodes, 1])
    plt.xlabel("Time")
    plt.ylabel("Force")
    savemyfig("Excitation_forces")

    trajs = [traj]

    count = counter()
    for traj in trajs:
        _t = traj[-2]
        _kepe = [tmodel.KEPE(traj[0][i], traj[1][i], graph_truss) for i in range(len(_t))]
        _ke = jnp.array([i[0] for i in _kepe])
        _pe = jnp.array([i[1] for i in _kepe])
        _t = traj[-2]

        fig, ax = panel(1, 1, dpi=70)
        
        plt.plot(_t, _ke, label="KE")
        plt.plot(_t, _pe, label="PE")
        plt.plot(_t, _ke + _pe, label="Ham")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc=(-0.1, 1.02), ncols=4)
        savemyfig(f"example_kepe_{next(count)}")



    count = counter()
    for traj in trajs:
        _t = traj[-2]
        X = traj[0]
        fig, axs = panel(1, 2, figsize=(8, 4), dpi=70, t_p=0.15)
        fig.suptitle('Position')
        axs[0].plot(_t, X[:, :, 0], label="x")
        axs[0].set_xlabel("Time")
        # axs[0].legend()
        axs[1].plot(_t, X[:, :, 1], label="y")
        axs[1].set_xlabel("Time")
        # axs[1].legend()
        savemyfig(f"example_position_{next(count)}")

    count = counter()
    for traj in trajs:
        X = traj[2]
        fig, axs = panel(1, 2, figsize=(8, 4), dpi=70, t_p=0.15)
        fig.suptitle('Acceleration', fontsize=16)
        axs[0].plot(_t, X[:, :, 0], label="x")
        # axs[0].legend()
        axs[1].plot(_t, X[:, :, 1], label="y")
        # axs[1].legend()
        savemyfig(f"example_acc_{next(count)}")

    # anim = pt.plot_truss_animation([traj], 
    #                             graph_truss, 
    #                             index=(0, -1, 10),
    #                             mkplot=dict(y=f_nodes), 
    #                             ylim=(-1, 1), 
    #                             support=(0, 1, 2, 3), 
    #                             view="x",
    #                             x_scale=20,
    #                             legend_prop=dict(fontsize=10, 
    #                                              loc=3, 
    #                                              bbox_to_anchor=(0, 1), 
    #                                              ncols=100)
    #                            )
    # plot_tools.savehtml5(anim, f_obj.fig("example_animation.html"), mp4=True)


    ##############################################################################
    # Generate data
    ##############################################################################

    generate_traj = data_gen.get_gen_traj(tmodel, state,
                                        constraints=constraints, 
                                        constant_mass_inv=M_1, 
                                        use_dissipative_forces=False)
    def generate_data(*args, **kwargs):
        traj = generate_traj(*args, **kwargs)
        return traj[2].reshape(-1, N*DIM)

    f_dirs = [
            {"x": f_nodes},
            {"y": f_nodes},
            {"z": f_nodes},
        ][:DIM]

    EXTERNAL_FORCE = data_gen.getF(f_dirs, 
                                F0, nodes.position.shape)

    fig, ax = panel(1, 1)
    for x in EXTERNAL_FORCE.T:
        plt.bar(range(len(x)), x)
    savemyfig("data_excitation_force")

    _RUNS = 50
    _ACCELERATION = vmap(lambda x: generate_data(x, runs=_RUNS, stride=STRIDE, rate=rate))(EXTERNAL_FORCE.T)

    ##############################################################################
    # Plot generated data
    ##############################################################################

    count = counter()
    with plt.rc_context({"figure.figsize": (6, 3)}):
        for j in range(len(_ACCELERATION)):
            fig, ax = panel(1, 1)
            for i in f_nodes:
                for k in range(DIM):
                    plt.plot(_ACCELERATION[j].reshape(-1, N, DIM)[:, i, k], c=pt.colors[k])
            plt.title(f"Force case {j+1}")
            savemyfig(f"data_accelertion_case_{next(count)}")


    ##############################################################################
    # Add noise to data and plot
    ##############################################################################
    ACCELERATION = utils.add_noise(0.0, _ACCELERATION)

    # np.savetxt("../dataset/Truss_DI/data1/EXTERNAL_FORCE.txt", np.array(EXTERNAL_FORCE))
    # np.savetxt("../dataset/Truss_DI/data1/ACCELERATION.txt", np.array(ACCELERATION))
    # np.savetxt("../dataset/Truss_DI/data1/description.txt", np.array([str(truss)]), fmt="%s")
    # np.savetxt("../dataset/Truss_DI/data1/nodes_position.txt", np.array(nodes.position))
    # np.savetxt("../dataset/Truss_DI/data1/nodes_mass.txt", np.array(graph_truss.nodes.nodal_mass))
    # np.savetxt("../dataset/Truss_DI/data1/edges_index.txt", np.array([senders, receivers], dtype=np.int64), fmt="%d")

    count = counter()
    with plt.rc_context({"figure.figsize": (6, 3)}):
        for j in range(len(_ACCELERATION)):
            fig, ax = panel(1, 1, dpi=70)
            for i in f_nodes:
                for k in range(DIM):
                    plt.plot(_ACCELERATION[j].reshape(-1, N, DIM)[:, i, k], c=pt.colors[k], alpha=0.4)
                    plt.plot(ACCELERATION[j].reshape(-1, N, DIM)[:, i, k], "--", c=pt.colors[k])
            plt.title(f"Force case {j+1}")
            savemyfig(f"noise_data_accelertion_case_{next(count)}")
        

    ##############################################################################
    # Define an ML model
    ##############################################################################

    N = len(nodes.position)
    Ne = edges.type.shape[-1]
    EA_fn = lambda d: EA0 * (1 - d)
    model = models.TrussGraphModel(key, N=N, Ne=Ne, dim=DIM, 
                                runs=_RUNS, stride=STRIDE, 
                                constraints=constraints, 
                                EA_fn=EA_fn, 
                                NODAL_MASS0=0.0, 
                                M_1=None,
                                activation=jax.nn.leaky_relu, 
                                hidden_dim=128,
                                latent_dim=8,
                                )



    ##############################################################################
    # Compare EA and damage on untrained model
    ##############################################################################

    pathid = f_obj.fig("untrained_EA")    
    pt.compareEA(model, graph_truss, EA_fn, EAd, pathid=pathid)
    savemyfig("untrained_EA")

    pathid = f_obj.fig("untrained_Damage")    
    pt.compareDamage(model, graph_truss, dd, pathid=pathid)
    savemyfig("untrained_Damage")


    ##############################################################################
    # Run untrained model and plot results
    ##############################################################################

    STATE = (1*R, 0*R, 0*R, graph_truss, 0.0, dt)

    model = models.copy_model(model, runs=_RUNS)
    ind = min(DIM-1, 2)
    _dim = min(DIM-1, 2)

    traj_data = generate_traj(EXTERNAL_FORCE[:, ind], runs=_RUNS, rate=rate, stride=STRIDE)
    traj_em = model((STATE, EXTERNAL_FORCE[:, ind], rate), train=False)
    traj_um = model((STATE, EXTERNAL_FORCE[:, ind], rate))

    As = ACCELERATION[ind][:_RUNS].reshape(-1, N, DIM)
    _t = traj_data[-2]

    with plt.rc_context(pt.small_ticks(0.5)):
        _n = len(f_nodes)
        fig, axs = plt.subplots(_n, 1, figsize=(12, 2*_n), sharex=True, dpi=70)

        axs = np.array(axs)

        for ax,i in zip(axs.ravel(), f_nodes):

            c1 = pt.colors[0]
            c2 = pt.colors[1]
            
            ax.scatter(_t, traj_data[2][:, i, _dim], c=c1, marker="x", s=20, label="Training data")
            ax.plot(_t, traj_em[2][:, i, _dim], "-", c=c1, label="Exact model")

            ax.scatter(_t, As[:, i, _dim], marker="o", fc="none", ec="red", s=20, label="Training data+noise")
            ax.plot(_t, traj_um[2][:, i, _dim], "-", c=c2, label="Untrained model")
            ax.set_ylabel("$A_y^{" + str(i+1) + "}$")
        ax.legend(ncols=4, loc=2, bbox_to_anchor=(0, 2.5), fontsize=14)
        ax.set_xlabel("Time")
        savemyfig("untrained_results")


    ##############################################################################
    # Copy model and define loss function
    ##############################################################################
    updated_model = model
    data = [((STATE, EXTERNAL_FORCE[:, i], rate), ACCELERATION[i]) for i in range(len(ACCELERATION))]


    def mse(y, y_, M=None, scale=1.0): 
        dy = (y - y_)
        A = jnp.mean(jnp.square(dy), axis=0) / scale
        if M is None:
            return A.mean()
        return (M @ A).mean() / M.mean()
        
    def loss(model, x, y):
        σ = jnp.std(y, axis=0)
        y = y[:_RUNS, :]
        y_ = model(x)[2].reshape(_RUNS, -1)
        L1 = mse(y, y_, M=M, scale=σ) 
        return L1 


    print("LOSS:", loss(model, (STATE, EXTERNAL_FORCE[:, 0], rate), ACCELERATION[0]))

    ##############################################################################
    # Define time length, sensor locations, optim and train function
    ##############################################################################

    import optax
    trN = 50

    np.random.seed(SSEED)
    ch_ = np.random.choice(len(node_position)-4, nsensors, replace=False) + 4
    ch_ = sum([[3*i, 3*i+1, 3*i+2] for i in ch_], start=[])
    # sensors = jnp.array([12, 13, 14, 18, 19, 20])
    sensors = jnp.array(ch_)
    print("Sensors: ", sensors)

    def loss(model, x, y):
        σ = jnp.std(y, axis=0)
        y = y[:trN, :]
        y_ = model(x)[2].reshape(trN, -1)
        L1 = mse(y[:trN][:, sensors], y_[:trN][:, sensors], 
                M=None, scale=σ[sensors])
        return L1 

    lr = 1.0e-3
    optim = optax.adam(lr)

    updated_model = models.copy_model(updated_model, runs=trN)
    _train = train.train(loss=loss)


    ##############################################################################
    # Train model
    ##############################################################################
    config.update("jax_debug_nans", True)
    updated_model = _train(updated_model, data, [], optim, 1000, 10)


    ##############################################################################
    # Compare EA and damage of trained model
    ##############################################################################


    fig, axs = panel(3, 1, dpi=70, figsize=(8, 12), vs=0.4)

    pathid = f_obj.fig("trained_EA")
    pt.compareEA(updated_model, graph_truss, EA_fn, EAd, 
                    epsilon=0.0, ax=axs[:2], pathid=pathid)

    pathid = f_obj.fig("trained_Damage")
    pt.compareDamage(updated_model, graph_truss, dd, 
                    epsilon=1.0, ax=axs[2], pathid=pathid)

    # pt.compareOffset(updated_model, graph_truss, offset, 
    #                  epsilon=None, ax=axs[3])

    axs[1].legend(loc=4, ncols=2, bbox_to_anchor=(1, 1))
    axs[2].legend(loc=4, ncols=2, bbox_to_anchor=(1, 1))
    savemyfig("trained_EA_Damage")

    ##############################################################################
    # 
    ##############################################################################
    # MLP_z = vmap(lambda x: utils.MLP(x, updated_model.layers[1]))

    # def getfori(i):
    #     dx = jnp.array([[i]]*Ne).reshape(Ne, 1)
    #     xz = jnp.hstack((edges.type, dx))
    #     return (MLP_z(xz) + jnp.square(dx))[-3]

    # dx = jnp.array([i/1e6 for i in range(1000)])
    # plt.plot(dx, vmap(getfori)(dx).reshape(-1))


    ##############################################################################
    # Extrapolation and plotting 
    ##############################################################################
    ttN = 4*trN

    ind = min(DIM-1, 2)
    _dim = min(DIM-1, 2)
    traj = generate_traj(EXTERNAL_FORCE[:, ind], runs=ttN, stride=STRIDE, rate=rate)
    mtraj = models.copy_model(updated_model, runs=ttN)((STATE, EXTERNAL_FORCE[:, ind], rate))

    pathid = f_obj.fig("trained_results_sensors")
    fig, axs = plt.subplots(len(sensors), 1, figsize=(10, 2*len(sensors)), sharey=True, sharex=True, dpi=70)
    for ax,s in zip(axs.ravel(), np.sort(sensors)):
        i, dim = s // 3, s % 3

        y = traj[2][:, i, dim]
        t_ = range(len(y))
        pt.plotobj(f"actual_sensor_{s}", "plot", t_, y, metadata={"lw":2, "c":"r", "alpha":1}, ax=ax, path=pathid).pt()
        y = mtraj[2][:, i, dim]
        pt.plotobj(f"ml_sensor_{s}", "plot", t_, y, metadata={"ls": "--", "c":"k", }, ax=ax, path=pathid).pt()

        # ax.plot(traj[2][:, i, dim], lw=2, c="r", alpha=1)
        # ax.plot(mtraj[2][:, i, dim], "--", c="k")

        ax.axvspan(0, trN, alpha=0.2, color="b")
        ax.text(10, -0.1, "Training Region",  verticalalignment='top', fontsize=12)
        ax.text(60, -0.1, "Extrapolation",  verticalalignment='top', fontsize=12)
        ax.set_ylabel(r"$A_{"+f"{dim}"+"}^{" + f"{i+1}"+"}$")
    ax.set_xlabel("Time (ms)")
    savemyfig("trained_results_sensors")


    all_sensors = np.array(range(3*len(node_position)))
    supports = np.array(range(3*4))
    rsensors = set(all_sensors) - set([int(i) for i in sensors]) - set(supports)

    # fig, axs = plt.subplots(len(rsensors), 1, figsize=(10, 2*len(rsensors)), sharey=True, sharex=True, dpi=70)
    # for ax,s in zip(axs.ravel(), sorted(rsensors)):
    #     i, dim = s // 3, s % 3
    #     ax.plot(traj[2][:, i, dim], lw=2, c="r", alpha=1)
    #     ax.plot(mtraj[2][:, i, dim], "--", c="k")
    #     ax.axvspan(0, trN, alpha=0.2, color="b")
    #     ax.text(10, -0.1, "Training Region",  verticalalignment='top', fontsize=12)
    #     ax.text(60, -0.1, "Extrapolation",  verticalalignment='top', fontsize=12)
    #     ax.set_ylabel(r"$A_{"+f"{dim}"+"}^{" + f"{i+1}"+"}$")
    # ax.set_xlabel("Time (ms)")
    # savemyfig("trained_results_remaining_nodes")


    # anim = pt.plot_truss_animation([traj_em, traj_um], 
    #                             graph_truss, 
    #                             index=(0, -1, 9),
    #                             mkplot=dict(y=f_nodes), 
    #                             ylim=(-0.2, 0.2), 
    #                             support=(0, 1, 2, 3), 
    #                             view="x",
    #                             x_scale=20,
    #                             legend_prop=dict(fontsize=10, 
    #                                              loc=3, 
    #                                              bbox_to_anchor=(0, 1), 
    #                                              ncols=100)
    #                            )
    # anim

    # anim = pt.plot_truss_animation([traj, mtraj], 
    #                             graph_truss, 
    #                             index=(0, -1, 9),
    #                             mkplot=dict(y=f_nodes), 
    #                             ylim=(-0.2, 0.2), 
    #                             support=(0, 1, 2, 3), 
    #                             view="x",
    #                             x_scale=20,
    #                             legend_prop=dict(fontsize=10, 
    #                                              loc=3, 
    #                                              bbox_to_anchor=(0, 1), 
    #                                              ncols=100)
    #                            )
    # anim


# SSEEDs = [2020, 213, 435, 546]
# nsensors_s = [2, 3, 4]

# DSEED = 2020
# ndamage = 4
# fdamage = 1
# Fnodes = [8, 9]
# NAME = "EXAMPLE2"

# for SSEED in SSEEDs:
#     for nsensors in nsensors_s:
#         main(NAME, DSEED, ndamage, fdamage, Fnodes, SSEED, nsensors)


# SSEED = 2424
# nsensors = 2

# DSEEDs = [2020, 2323, 341, 3242]
# ndamages = [2, 4, 6, 8]
# fdamages = [1, 2]

# for DSEED in DSEEDs:
#     for ndamage in ndamages:
#         for fdamage in fdamages:
#             main(NAME, DSEED, ndamage, fdamage, Fnodes, SSEED, nsensors)


