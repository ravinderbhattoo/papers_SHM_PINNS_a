import shadow
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import jax.numpy as jnp
from .utils import index_traj, getEA, getDamage, getOffset
from .io import check

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150

plot_size = {
    'figure.figsize':(6, 6),
    'figure.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 1.3,
}

plt.rcParams.update(plot_size)

def small_ticks(s):
    return {
        'xtick.major.size': 10*s,
        'ytick.major.size': 10*s,
        'xtick.minor.size': 8*s,
        'ytick.minor.size': 8*s,
    }

colors = ["#3f90da", "#ffa90e", "#94a4a2",
                "#832db6", "#a96b59", "#e76300", "#b9ac70",
                "#717581", "#92dadd", "#bd1f01"]
prop_cycle = {
    'axes.prop_cycle': mpl.cycler(color=colors)
}

plt.rcParams.update(prop_cycle)

def savehtml5(anim, filename, mp4=False, mp4_kwargs={}):
    video = anim.to_html5_video()
    with open(f"../output/anim/{filename}.html", "w+") as f:
        f.write(video)
    if mp4:
        savemp4(anim, f"../output/anim/{filename}", **mp4_kwargs)

def savemp4(anim, filename, fps=10):
    anim.save(f'{filename}.mp4', writer="ffmpeg")

def plot_truss_animation(_datas, graph, index=None, external_force=None, pstyle=None, mkplot=None, mkstyle=None,
                         ylim=(-5, 5), support=None, view="z", x_scale=10,
                         legend_prop=dict(bbox_to_anchor=(0, 1), loc=3, ncols = 6),
                         xlabel=None,
):

    n = len(_datas)

    _t = _datas[0][-2]
    
    if index is not None:
        datas = [index_traj(traj, index, shift=0.0)
                  for traj in _datas]

    if mkplot is not None:
        _mkplot = {"x": jnp.array([]),
                   "y": jnp.array([]),
                   "z": jnp.array([])}
        _mkplot.update(mkplot)
        _mkplot["x"] = jnp.array(_mkplot["x"])
        _mkplot["y"] = jnp.array(_mkplot["y"])
        _mkplot["z"] = jnp.array(_mkplot["z"])
        mkplot = _mkplot

    if mkstyle is None:
        mkstyle = [{"ls":"-", "alpha":0.8}, {"ls": "--", "alpha":0.8}, {"ls": "."}][:n]


    if pstyle is None:
        pstyle = [{'lw': 4}]
        for i in range(n-1):
            pstyle += [{}]

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib import animation
    from IPython.display import HTML

    ani_style = {
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 100,
        "animation.html": "jshtml",
    }

    with plt.rc_context(ani_style):

        def view_apply(R):
            R_ = np.array(R)
            if view=="x":
                R_[:, :, 1] = R[:, :, 2]
                R_[:, :, 0] = R[:, :, 1]
            elif view=="y":
                R_[:, :, 1] = R[:, :, 2]
            return R_

        ts = [data[-2] for data in datas]
        Rs = [view_apply(data[0]) for data in datas]
        As = [data[2] for data in _datas]

        t = ts[0]

        if external_force is not None:
            EXF = vmap(lambda x: external_force(None, None, x, None))(t.reshape(-1))

        xlo, xhi = min([R[:, :, 0].min() for R in Rs]), max([R[:, :, 0].max() for R in Rs])
        ylo, yhi = min([R[:, :, 1].min() for R in Rs]), max([R[:, :, 1].max() for R in Rs])

        xl = xhi - xlo
        yl = yhi - ylo

        xlo -= xl*0.1
        xhi += xl*0.1

        ylo -= yl*0.1
        yhi += yl*0.1

        if mkplot is not None:
            fig, axs = plt.subplots(2, 1, dpi=300)
            ax, ax2 = axs

            ax2.minorticks_on()
            for axis in ['top', 'bottom', 'left', 'right']:
                ax2.spines[axis].set_color('lavender')
            ax2.grid(which = "major", linewidth = 1)
            ax2.grid(which = "minor", linewidth = 0.2)


            ax2.axis([0, _t[-1], *ylim])
        else:
            fig, ax = plt.subplots(1, 1, dpi=300)

        ax.axis('off')
        ax.axis([xlo, xhi, ylo, yhi])
        ls = [ax.plot([],[], **pstyle[i])[0] for i in range(n)]

        if support is not None:
            support = np.array(support)
            x = Rs[0][0, support, 0]
            y = Rs[0][0, support, 1]
            ax.scatter(x, y, marker="^", c="k", s=80, zorder=100)

        if mkplot is not None:
            ap = mkplot["x"]
            bp = mkplot["y"]
            cp = mkplot["z"]
            nn = len(ap) + len(bp) + len(cp)
            ls2 = []
            for i in range(n):
                plt.gca().set_prop_cycle(None)
                ls2 += [[ax2.plot([], [], **mkstyle[i])[0] for j in range(nn)]]
            legends = []
            legends += ["$A_x^{"+str(i.item()+1)+"}$" for i in ap]
            legends += ["$A_y^{"+str(i.item()+1)+"}$" for i in bp]
            legends += ["$A_z^{"+str(i.item()+1)+"}$" for i in cp]
            ax2.legend(legends, **legend_prop)
            ax2.set_ylabel("Acceleration")
            if xlabel==None:
                ax2.set_xlabel("Time")
            else:
                ax2.set_xlabel(xlabel)
        else:
            ls2 = [None]*len(ls)

        def get_xy(XY, i):
            x = []
            y = []
            for r,s in zip(graph.receivers, graph.senders):
                x += [XY[r, 0], XY[s, 0], np.nan]
                y += [XY[r, 1], XY[s, 1], np.nan]
            return x, y

        def set_anim(l, l2, R, A, i):
            XY = R[0] + x_scale*(R[i] - R[0])
            x, y = get_xy(XY, i)
            l.set_data(x, y)
            if mkplot is not None:
                abc = []
                if ap.shape[0]>0:
                    abc += [A[:i*index[-1], ap, 0]]
                if bp.shape[0]>0:
                    abc += [A[:i*index[-1], bp, 1]]
                if cp.shape[0]>0:
                    abc += [A[:i*index[-1], cp, 2]]
                ab = jnp.hstack(abc)
                for ind, l2_ in enumerate(l2):
                    l2_.set_data(_t[:len(ab)], ab[:, ind])

        def animate(i):
            for l, l2, R, A in zip(ls, ls2, Rs, As):
                set_anim(l, l2, R, A, i)

        ani = animation.FuncAnimation(fig, animate, frames=len(t))
        plt.close()
    return ani

def plotobjfromfile(filename, plt=plt, ax=None):
    with open(filename, 'rb') as fhandle:
        d = pickle.load(fhandle)
    return plotobj(d["name"], d["typeof"], d["x"], d["y"], metadata=d["metadata"], plt=plt, ax=ax, path=None)

class plotobj():
    def __init__(self, name, typeof, x, y, metadata={}, plt=plt, ax=None, path=None):
        self.plt = plt
        self.ax = ax
        _dict = dict(
            typeof = typeof,
            x = x,
            y = y,
            metadata = metadata,
            path = path,
            name = name,
        )
        self.data = _dict
    def pt(self, ):
        d = self.data
        if d["path"] is not None:
            filename = f"{d["path"]}/{d["name"]}.pk"
            check(filename)
            with open(filename, 'wb+') as fhandle:
                pickle.dump(d, fhandle, protocol=pickle.HIGHEST_PROTOCOL)
        x = d["x"]
        y = d["y"]
        metadata = d["metadata"]
        typeof = d["typeof"]
        if self.ax is None:
            return getattr(self.plt, typeof)(x, y, **metadata)
        else:
            return getattr(self.ax, typeof)(x, y, **metadata)
            

def compareEAh(model, graph, EA_fn, EAd, epsilon=None, ax=None, pathid=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax = ax[::-1]
    EAe = getEA(model, graph, EA_fn)
    a_ = 100 * (EAe - EAd) / EAd
    a = jnp.abs(a_)

    if epsilon is not None:
        a_ = jnp.where(a < epsilon, 0.0, a_)
        EAe = jnp.where(a < epsilon, EAd, EAe)

    a = jnp.abs(a_)
    
    x = range(1, len(a)+1)
    y = a

    plotobj("EA_error", "bar", x, y, ax=ax[0], path=pathid).pt()
    ax[0].set_xlabel("Member number")
    ax[0].set_ylabel(r"% error")

    x = jnp.array(range(1, len(a)+1))
    plotobj("EA_actual", "bar", x-0.2, EAd, metadata={"label": "Actual", "width":0.4}, ax=ax[1], path=pathid).pt()
    plotobj("EA_estimated", "bar", x+0.2, EAe, metadata={"label": "Estimated", "width":0.4}, ax=ax[1], path=pathid).pt()

    # ax[1].bar(x-0.2, EAd, width=0.4, label="Actual")
    # ax[1].bar(x+0.2, EAe, width=0.4, label="Estimated")

    ax[1].legend(loc=4, bbox_to_anchor=(1, 1), ncol=2)
    ax[1].set_xlabel("Member number")
    ax[1].set_ylabel("EA")

    return (ax[0].get_figure(), ax), (EAd, EAe, epsilon)

def compareEA(model, graph, EA_fn, EAd, epsilon=None, ax=None, pathid=None):
    if ax is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw=dict(hspace=0.0), sharex=True)
    ax2, ax1 = axs
    ax1.invert_yaxis()
    ax1.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    # ax2.xaxis.tick_top()

    EAe = getEA(model, graph, EA_fn)
    a_ = 100 * (EAe - EAd) / EAd
    a = jnp.abs(a_)

    if epsilon is not None:
        a_ = jnp.where(a < epsilon, 0.0, a_)
        EAe = jnp.where(a < epsilon, EAd, EAe)

    a = jnp.abs(a_)
    
    x = range(1, len(a)+1)
    y = a

    plotobj("EA_error", "bar", x, y, ax=ax1, path=pathid,  metadata={"hatch":"//", "label":"Error"} ).pt()
    ax1.set_xlabel("Member Number")
    ax1.set_ylabel(r"Error (%)")

    x = jnp.array(range(1, len(a)+1))
    plotobj("EA_actual", "bar", x-0.2, EAd, metadata={"label": "Actual", "width":0.4, "label":"Actual"}, ax=ax2, path=pathid).pt()
    plotobj("EA_estimated", "bar", x+0.2, EAe, metadata={"label": "Estimated", "width":0.4, "label":"Estimated"}, ax=ax2, path=pathid).pt()
    ax2.set_ylabel("EA")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=4)
    
    return (ax1.get_figure(), ax), (EAd, EAe, epsilon)


def compareDamage(model, graph, dd_, epsilon=None, ax=None, pathid=None):
    de = getDamage(model, graph) * 100
    dd = dd_ * 100
    if epsilon is not None:
        de = jnp.where(jnp.abs(de) < epsilon, 0.0, de)
    x = jnp.array(range(1, len(dd)+1))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    plotobj("Damage_actual", "bar", x-0.2, dd, metadata={"label": "Actual", "width":0.4}, ax=ax, path=pathid).pt()
    plotobj("Damage_estimated", "bar", x+0.2, de, metadata={"label": "Estimated", "width":0.4}, ax=ax, path=pathid).pt()

    ax.legend()
    ax.set_ylabel("Damage value (%)")
    ax.set_xlabel("Member number")
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    return ", ".join([f"({i:.4f}, {j:.4f}, Δ:{(i-j):.4f})" for i,j in zip(dd, de)])

def compareOffset(model, graph, dd, epsilon=None, ax=None):
    de = getOffset(model, graph)
    if epsilon is not None:
        de = jnp.where(de < epsilon, 0.0, de)
    x = jnp.array(range(1, len(dd)+1))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x-0.2, dd, width=0.4, label="Actual")
    ax.bar(x+0.2, de, width=0.4, label="Estimated")

    ax.legend()
    ax.set_ylabel("Offset value")
    ax.set_xlabel("Member number")
    return ", ".join([f"({i:.4f}, {j:.4f}, Δ:{(i-j):.4f})" for i,j in zip(dd, de)])

