import numpy as np
import matplotlib.pyplot as plt

TEXT_KWARGS = {
                "fontsize": 4,
                "backgroundcolor": (1, 1, 1, 0.9),
                "c": "k",
                "bbox": dict(
                boxstyle="square",
                ec="r",
                lw=0.4,
                fc="w",
                alpha=0.8,
                )
            }

TEXT_KWARGS2 = {
                "fontsize": 4,
                "backgroundcolor": (1, 1, 1, 0.9),
                "c": "k",
                "bbox": dict(
                boxstyle="square",
                ec="none",
                fc="w",
                alpha=0.8,
                )
            }

class Truss():
    def __init__(self, nodes, edges, cross_section_area=None, name="Truss", support=None):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.nedges = len(edges["index"])
        self.support = support
        self.nnodes, self.dim = self.nodes["position"].shape

        if cross_section_area is None:
            self.edges["A"] = np.array([1.0]*self.nedges)
        else:
            self.edges["A"] = cross_section_area
        self.edges["E"] = np.array([1.0]*self.nedges)

    def __repr__(self, ):
        return self.name

    def set_properties(self, E=1.0, A=1.0):
        if isinstance(E, float):
            self.edges["E"] = [E]*self.nedges
        else:
            self.edges["E"] = E
        if isinstance(A, float):
            self.edges["A"] = [A]*self.nedges
        else:
            self.edges["A"] = A

    def displacement(self, F, dofs=None):
        # Ku = F
        self.u = np.zeros((len(self.K), 1))
        if dofs is None:
            self.u = np.linalg.pinv(self.K).dot(F)

        else:
            self.u[dofs] = np.linalg.pinv(self.K[dofs, :][:, dofs]).dot(F[dofs])[:]
        return self.u

    def stress(self, ):
        pos = self.nodes["position"]
        pos2 = pos + self.u.reshape(*pos.shape)
        self.sigma = np.zeros(len(self.edges["A"]))
        self.delta = 0.0*self.sigma
        E, A = self.edges["E"], self.edges["A"]
        for k, e in enumerate(self.edges["index"]):
            i, j = e
            dx1 = pos[j] - pos[i]
            l1 = (sum((dx1)**2))**0.5

            dx2 = pos2[j] - pos2[i]
            l2 = (sum((dx2)**2))**0.5
            self.sigma[k] = E[k] * (l2 - l1) / l1
            self.delta[k] = l2 - l1
        self.T = self.sigma * self.edges["A"]
        return self.sigma

    def cal_euler_delta(self):
        return self.edges["kappa"] * self.edges["A"]

    def cal_prelim(self):
        dim = self.dim
        pos = self.nodes["position"]
        self.edges["L"] = np.array(self.edges["A"]) * 0.0
        for k, e in enumerate(self.edges["index"]):
            i, j = e
            index = np.array([i*dim+p  for p in range(dim)] +
                            [j*dim+p  for p in range(dim)])
            dx = pos[j] - pos[i]
            l = (sum((dx)**2))**0.5
            self.edges["L"][k] = l
        self.edges["kappa"] = np.pi / self.edges["L"]

    def K_matrix(self):
        dim = self.dim
        N = dim * len(self.nodes["position"])
        self.K = np.zeros((N, N))
        E, A = self.edges["E"], self.edges["A"]
        pos = self.nodes["position"]
        for k, e in enumerate(self.edges["index"]):
            i, j = e
            index = np.array([i*dim+p  for p in range(dim)] +
                            [j*dim+p  for p in range(dim)])
            dx = pos[j] - pos[i]
            l = (sum((dx)**2))**0.5
            cs = dx / l
            K = E[k]*A[k] / l
            K = K * np.array([[i*j for i in cs] for j in cs])
            K = np.vstack((np.hstack([K, -K]), np.hstack([-K, K])))
            for ind, kk in enumerate(index):
                self.K[index, kk] += K[:, ind]
        return self.K

    def connection_matrix(self,):
        pos = self.nodes["position"]
        n = len(pos)
        M = np.zeros((n,n), dtype=np.bool_)
        iedges = self.edges["index"]
        for i,j in iedges:
            M[i, j] = True
            M[j, i] = True
        self.M = M
        return M

    def plot(self, ax=None,
                 scatter_kwargs={"s": 0},
                 plot_kwargs={"c": "k"},
                 damage_plot_kwargs=None,
                 text_kwargs={},
                 text_kwargs2={},
                 support_kwargs=dict(marker="^", c="r"),
                 ThreeD=False,
                 axis="off",
                 displacement = 0.0,
                 **kwargs,
            ):
        _text_kwargs = TEXT_KWARGS
        _text_kwargs.update(text_kwargs)
        _text_kwargs2 = TEXT_KWARGS2
        _text_kwargs2.update(text_kwargs2)
        pos = self.nodes["position"] + displacement
        if damage_plot_kwargs is not None:
            damage_ind = np.array(damage_plot_kwargs.pop("members"))
            damage_val = np.array(damage_plot_kwargs.pop("values"))
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=300)

        if ThreeD:
            ax = plt.axes(projection='3d')
            ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], **scatter_kwargs, **kwargs)
            ax.scatter3D(pos[self.support, 0], pos[self.support, 1], pos[self.support, 2], **support_kwargs)
            for e in self.edges["index"]:
                ax.plot3D(pos[e, 0], pos[e, 1], pos[e, 2], **plot_kwargs, **kwargs)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], **scatter_kwargs, **kwargs)
            for i in range(pos.shape[0]):
                ax.text(pos[i, 0], pos[i, 1], i+1, **_text_kwargs)
            ax.scatter(pos[self.support, 0], pos[self.support, 1], **support_kwargs)
            for ind, e in enumerate(self.edges["index"]):
                ax.plot(pos[e, 0], pos[e, 1], **plot_kwargs, **kwargs)
                x = pos[e, 0][0]*0.6 + pos[e, 0][1]*0.4
                y = pos[e, 1][0]*0.6 + pos[e, 1][1]*0.4
                ax.text(x, y, ind+1, **_text_kwargs2)
                if damage_plot_kwargs!= None:
                    if ind in damage_ind:
                        ax.plot(pos[e, 0], pos[e, 1], **damage_plot_kwargs, **kwargs)
                        if (damage_val != None).any():
                            dval_txt = f"{damage_val[np.argwhere(damage_ind==ind)[0][0]]:.1f}%"
                            ax.text(x, y+0.2, dval_txt, **_text_kwargs2, ha="center")

        ax.set_aspect('equal')
        if axis=="off":
            ax.set_axis_off()
        # ax.set_title(self.name, **_text_kwargs)

        return ax.get_figure(), ax

def Truss_from_connctions(x, M, name=None):
    nodes = dict(position=x)
    n, n = M.shape
    iedges = []
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] > 0.5:
                iedges += [[i,j]]
    edges = dict(index=iedges)
    if name is None:
        name = "From connection matrix"
    return Truss(nodes, edges, name=name)




class Pratt(Truss):
    def __init__(self, length, height, segments=10, seed=0, rand_=0.0):
        if seed is not None:
            np.random.seed(seed)
        segments = (segments // 2) * 2
        segments += 1
        dx = length/(segments-1)

        pos_bnodes = [[i*dx+dx*np.random.randn()*rand_, 0.0] for i in range(segments-1)] + [[length, 0.0]]
        pos_tnodes = [[i[0], height] for i in pos_bnodes[1:-1]]

        pos_nodes = np.array(pos_bnodes + pos_tnodes)
        n_nodes = len(pos_nodes)

        v_iedges = [[i+1, segments+i] for i in range(segments-2)]
        s_iedges = [[0, segments], [len(pos_nodes)-1, segments-1]]

        b_iedges = [[i, i+1] for i in range(segments-1)]
        t_iedges = [[segments+i, segments+1+i] for i in range(segments-3)]

        d1_iedges = [[segments+i, i+2] for i in range(segments//2-1)]
        d2_iedges = [[n_nodes-1-i, segments-i-3] for i in range(segments//2-1)]

        iedges = np.array(v_iedges + s_iedges + b_iedges + t_iedges + d1_iedges + d2_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"PrattTruss (Span: {length :.2f}, Height: {height :.2f}, Segments: {segments-1 :.0f})", support=np.array([0, segments-1]))

class Howe(Truss):
    def __init__(self, length, height, segments=10, seed=0, rand_=0.0):
        if seed is not None:
            np.random.seed(seed)
        segments = (segments // 2) * 2
        segments += 1
        dx = length/(segments-1)
        pos_bnodes = [[i*dx+dx*np.random.randn()*rand_, 0.0] for i in range(segments-1)] + [[length, 0.0]]
        pos_tnodes = [[i[0], height] for i in pos_bnodes[1:-1]]
        pos_nodes = np.array(pos_bnodes + pos_tnodes)
        n_nodes = len(pos_nodes)

        v_iedges = [[i+1, segments+i] for i in range(segments-2)]
        s_iedges = [[0, segments], [len(pos_nodes)-1, segments-1]]

        b_iedges = [[i, i+1] for i in range(segments-1)]
        t_iedges = [[segments+i, segments+1+i] for i in range(segments-3)]

        d1_iedges = [[segments+i+1, i+1] for i in range(segments//2-1)]
        d2_iedges = [[n_nodes-2-i, segments-i-2] for i in range(segments//2-1)]

        iedges = np.array(v_iedges + s_iedges + b_iedges + t_iedges + d1_iedges + d2_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"HoweTruss (Span: {length :.2f}, Height: {height :.2f}, Segments: {segments-1 :.0f})", support=np.array([0, segments-1]))


class Warren(Truss):
    def __init__(self, length, height, segments=10, seed=0, rand_=0.0):
        if seed is not None:
            np.random.seed(seed)
        segments = (segments // 2) * 2
        segments += 1
        dx = length/(segments-1)
        pos_bnodes = [[i*dx+dx*np.random.randn()*rand_, 0.0] for i in range(segments-1)] + [[length, 0.0]]

        pos_tnodes = [[i[0]+dx/2, height] for i in pos_bnodes[:-1]]

        pos_nodes = np.array(pos_bnodes + pos_tnodes)
        n_nodes = len(pos_nodes)

        b_iedges = [[i, i+1] for i in range(segments-1)]
        t_iedges = [[segments+i, segments+1+i] for i in range(segments-2)]

        d1_iedges = [[segments+i, i] for i in range(segments-1)]
        d2_iedges = [[n_nodes-1-i, segments-i-1] for i in range(segments-1)]


        iedges = np.array(b_iedges + t_iedges + d1_iedges + d2_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"WarrenTruss (Span: {length :.2f}, Height: {height :.2f}, Segments: {segments-1 :.0f})", support=np.array([0, segments-1]))



class Bar10(Truss):
    def __init__(self, l1=1.0, l2=1.0, h=1.0):
        pos_bnodes = [[0.0, 0.0], [l1, 0.0], [l1+l2, 0.0]]
        pos_tnodes = [[0.0, h], [l1, h], [l1+l2, h]]

        pos_nodes = np.array(pos_bnodes + pos_tnodes)

        n_nodes = len(pos_nodes)

        b_iedges = [[0, 1], [1, 2]]
        t_iedges = [[3, 4], [4, 5]]
        v_iedges = [[1, 4], [2, 5]]
        d1_iedges = [[0, 4], [1, 5]]
        d2_iedges = [[3, 1], [4, 2]]


        iedges = np.array(b_iedges + t_iedges + d1_iedges + d2_iedges + v_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"10 Bar", support=np.array([0, 3]))


class BoxTruss(Truss):
    def __init__(self, n=2, width=1.0, height=1.0):
        pos_bnodes = [[i*width+0.0, 0.0] for i in range(n+1)]
        pos_tnodes = [[i*width+0.0, height] for i in range(n+1)]

        pos_nodes = np.array(pos_bnodes + pos_tnodes)

        n_nodes = len(pos_nodes)

        b_iedges = [[i, i+1] for i in range(n)]
        t_iedges = [[n+i+1, n+i+2] for i in range(n)]
        v_iedges = [[i, i+n+1] for i in range(n+1)]
        d1_iedges = [[i, i+n+2] for i in range(n)]
        d2_iedges = [[i+1, i+n+1] for i in range(n)]


        iedges = np.array(b_iedges + t_iedges + v_iedges + d1_iedges + d2_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"Box Truss {n}", support=np.array([0, n]))


class Bar200(Truss):
    def __init__(self, w=6.1, h1=3.66, h2=9.144):
        def get_node_line(n1, w1, origin=(0,0)):
            pos_anodes = [[i*w1+origin[0], origin[1]] for i in range(n1+1)]
            return pos_anodes

        pos_lines = sum([get_node_line(4, w, origin=(0, -i*h1)) if i%2==0 else get_node_line(8, w/2, origin=(0, -i*h1))
                            for i in range(11)], start=[])
        pos = pos_lines + [[w, -10*h1-h2], [3*w, -10*h1-h2]]

        N = len(pos)

        def get_edges_h(i, n):
            if n==4:
                s = i//2*14
            if n==8:
                s = 5 + i//2*14
            return [[s+i, s+i+1] for i in range(n)]

        def get_edges_v(i, n):
            if n==4:
                s = i//2*14
                return [[s+i, s+5+2*i] for i in range(5)]
            if n==8:
                s = 5 + (i-1)//2*14
                return [[s+2*i, s+9+i] for i in range(5)]

        def get_edges_d1(i, n):
            if n==4:
                s = i//2*14
                return [[s+i, s+5+2*i+1] for i in range(4)] + [[s+i+1, s+5+2*i+1] for i in range(4)]
            if n==8:
                s = 6 + (i-1)//2*14
                return [[s+2*i, s+8+i] for i in range(4)] + [[s+2*i, s+8+i+1] for i in range(4)]

        edge_end = [[N-7+i, N-2] for i in range(3)] + [[N-5+i, N-1] for i in range(3)]


        iedges = []
        for i in range(10):
            iedges_h = get_edges_h(i, 4) if (i%2==0) else get_edges_h(i, 8)
            iedges_v = get_edges_v(i, 4) if (i%2==0) else get_edges_v(i, 8)
            iedges_d = get_edges_d1(i, 4) if (i%2==0) else get_edges_d1(i, 8)
            iie = []
            for j in range(4):
                iie += [iedges_v[j], iedges_d[j], iedges_d[4+j]]
            iedges += iedges_h + iie + [iedges_v[-1]]

        iedges_h = get_edges_h(11, 4)
        iedges += iedges_h + edge_end

        iedges = np.array(iedges)
        nodes = dict(position = np.array(pos))
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"Bar 200", support=np.array([N-2, N-1]))


class Bar25(Truss):
    def __init__(self, a1=8.0, a2=3.0, h1=4.0, h2=4.0):
        pos_bnodes = [[-a1/2+i*a1, -a1/2+j*a1, 0.0] for i in range(2) for j in range(2)]
        pos_mnodes = [[-a2/2+i*a2, -a2/2+j*a2, h1] for i in range(2) for j in range(2)]
        pos_tnodes = [[-a2/2+i*a2, 0.0, h1+h2] for i in range(2)]

        pos_nodes = np.array(pos_bnodes + pos_mnodes + pos_tnodes)

        n_nodes = len(pos_nodes)

        l1_iedges = [[0, i] for i in [4, 5, 6]]
        l1_iedges += [[1, i] for i in [4, 5, 7]]
        l1_iedges += [[2, i] for i in [4, 7, 6]]
        l1_iedges += [[3, i] for i in [7, 5, 6]]

        l2_iedges = [[i, j] for i in [8, 9] for j in [4,5,6,7]]

        l3_iedges = [ [i, j] for i, j in zip([4, 5, 7, 6], [5, 7, 6, 4])] + [[8, 9]]

        iedges = np.array(l1_iedges + l2_iedges + l3_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"Bar25", support=np.array([0, 1, 2, 3]))


class DomeBar120(Truss):
    def __init__(self, r1=15.89, r2=12.04, r3=6.94, h1=3.0, h2=5.85, h3=7.0):
        pos_r1nodes = [[r1*np.cos(i), r1*np.sin(i), 0.0] for i in np.arange(0, 2*np.pi, np.pi/6)]
        n1 = len(pos_r1nodes)
        N1 = list(range(n1))
        pos_r2nodes = [[r2*np.cos(i), r2*np.sin(i), h1] for i in np.arange(0, 2*np.pi, np.pi/12)]
        n2 = len(pos_r2nodes)
        N2 = list(range(n1, n1+n2))
        pos_r3nodes = [[r3*np.cos(i), r3*np.sin(i), h2] for i in np.arange(0, 2*np.pi, np.pi/6)]
        n3 = len(pos_r3nodes)
        N3 = list(range(n1+n2, n1+n2+n3))

        pos_nodes = np.array(pos_r1nodes + pos_r2nodes + pos_r3nodes + [[0.0, 0.0, h3]])

        n_nodes = len(pos_nodes)

        l_iedges = [[N1[i], N2[i*2+j-1]] for j in range(3) for i in range(n1)]
        l_iedges += [[N2[i], N2[i+1]] for i in range(n2-1)] + [[N2[n2-1], N2[0]]]

        l_iedges += [[N3[i], N2[i*2+j-1]] for j in range(3) for i in range(n3)]
        l_iedges += [[N3[i], N3[i+1]] for i in range(n3-1)] + [[N3[n3-1], N3[0]]]

        l_iedges += [[N3[i], n1+n2+n3] for i in range(n3)]


        iedges = np.array(l_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"DomeBar120", support=np.array(range(len(pos_r1nodes))))


class DomeBar52(Truss):
    def __init__(self, r1=6, r2=4, r3=2, h1=2.5, h2=3.731, h3=5.995):
        pos_r1nodes = [[r1*np.cos(i), r1*np.sin(i), 0.0] for i in np.arange(0, 2*np.pi, np.pi/4)]
        n1 = len(pos_r1nodes)
        N1 = list(range(n1))
        pos_r2nodes = [[r2*np.cos(i), r2*np.sin(i), h1] for i in np.arange(0, 2*np.pi, np.pi/4)]
        n2 = len(pos_r2nodes)
        N2 = list(range(n1, n1+n2))
        pos_r3nodes = [[r3*np.cos(i), r3*np.sin(i), h2] for i in np.arange(0, 2*np.pi, np.pi/2)]
        n3 = len(pos_r3nodes)
        N3 = list(range(n1+n2, n1+n2+n3))

        pos_nodes = np.array(pos_r1nodes + pos_r2nodes + pos_r3nodes + [[0.0, 0.0, h3]])

        n_nodes = len(pos_nodes)

        l_iedges = [[N1[i], N2[i+j-1]] for j in range(3) for i in range(n1-1)]


        l_iedges += [[N1[n1-1], N2[-2]], [N1[n1-1], N2[-1]], [N1[n1-1], N2[0]]]

        l_iedges += [[N2[i], N2[i+1]] for i in range(n2-1)] + [[N2[n2-1], N2[0]]]

        l_iedges += [[N3[i], N2[i*2+j-1]] for j in range(3) for i in range(n3)]
        l_iedges += [[N3[i], N3[i+1]] for i in range(n3-1)] + [[N3[n3-1], N3[0]]]

        l_iedges += [[N3[i], n1+n2+n3] for i in range(n3)]


        iedges = np.array(l_iedges)

        nodes = dict(position=pos_nodes)
        edges = dict(index=iedges)

        super().__init__(nodes, edges, name=f"DomeBar52", support=np.array(N1))



