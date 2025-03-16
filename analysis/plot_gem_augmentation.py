import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pooling.nn.gem_pool import GeneralizedMean, KolmogorovMean


def plot_means(
        path,
        p=None,
        n=101,
        k=100,
        d=2,
        seed=0,
        mu=0.25,
        std=1.0,
        xlim=(-3,3), 
        ylim=(-3,3),
        M_f=True,
        bias=np.array([0.2,-0.4]),
        legend_loc="lower right",
    ):
    ## TEST DATA
    np.random.seed(seed)
    X = torch.Tensor(np.random.normal(loc=mu, scale=std, size=(k,d)))

    ## COMPUTE p
    if p is None:
        p = torch.arange(-1000, 1001, 1, dtype=torch.float32)

    ## COMPUTE MEANS
    mean_g = []
    mean_k = []
    for p_ in p:
        gem = GeneralizedMean(p=p_)
        kom = KolmogorovMean(p=p_, b=0.9)
        mean_g.append(gem(X))
        mean_k.append(kom(X))
    mean_g = torch.stack(mean_g)
    mean_k = torch.stack(mean_k)

    ## EXAMINE DATA
    COLORS = {
        "X": "#aeafb0", # "blue",
        "M_p": "#ff3e3e", # "red",
        "M_f": "#669eff", # "green",
        "g": "#0bd100", # "green",
        "p": "#ffa200", # "orange",
    }
    size = 5
    fig, ax = plt.subplots(figsize=(5,4), dpi=500, nrows=1, ncols=1)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.1)
    
    ## PLOT DATA
    scatter_X = ax.scatter(*X.T, s=size, alpha=0.7, color=COLORS["X"], label="X")
    
    ## LABEL 10 POINTS FROM p
    idxs = np.array([0, 900, 950, 1001, 1050, 1100, -1])
    p_points = mean_k[idxs] if M_f else mean_g[idxs]
    p_labels = p[idxs]
    if len(p_labels.shape) == 1:
        p_labels = torch.tile(p_labels, (2,1)).T
    p_labels = [f"[{int(p[0])}, {int(p[1])}]" for p in p_labels]
    _ = ax.scatter(*p_points.T, alpha=1.0, s=30, c=COLORS["p"], label="p")
    for i in range(len(p_labels)):
        _ = ax.text(
            *(p_points[i] + bias),
            p_labels[i],
            fontsize=7,
            fontweight="heavy",
            c=COLORS["p"],
            bbox=dict(
                facecolor="w",
                ec="w", #COLORS["p"], 
                alpha=0.7,
                boxstyle='round',
                pad=0.
            ),
        )

    ## PLOT MEANS
    if M_f:
        kom_line = ax.plot(*mean_k[::5].T, "o-", alpha=0.3, lw=1, markersize=2, c=COLORS["M_f"], label="M_f")
    else:
        gem_line = ax.plot(*mean_g.T, "o-", alpha=0.3, lw=1, markersize=2, c=COLORS["M_p"], label="M_p")
    
    # ax.legend(loc="upper right")
    ax.set_title(f"μ = {mu}     σ = {std}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(frameon=True, loc=legend_loc) 
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()
    del fig
    return


def zelu(p, p_min=1e-4):
    p[p==0] = p_min
    return p
    # # p[abs(p) < p_min] = p_min
    # if abs(p) < p_min:
    #     return p_min
    # return p + p_min

def symexp(p):
    return np.sign(p) * (np.exp(np.abs(p))-1)

def tanh(p, p_max=5e+4):
    return p_max * np.tanh(0.005 * p)
    # return np.clip((10*p)**3, a_min=-p_max, a_max=p_max)
    # return np.clip(symexp(10*p), a_min=-p_max, a_max=p_max)


if __name__ == "__main__":

    v = 3.5
    p = torch.linspace(-v, v, steps=int(v*2*100), dtype=torch.float32)
    p = torch.round(zelu(tanh(p)), decimals=0)

    ## SYMMETRIC P
    k = 2.5
    mu=0.25
    p = zelu(torch.arange(-1000, 1001, 1, dtype=torch.float32))
    plot_means(
        path="/home/brothag1/code/stan/pooling/analysis/gem-jagged-sym-M_p.png",
        p=p,
        mu=mu,
        seed=11,
        M_f=False,
        xlim=(-2.75,3.25),
        ylim=(-2.75,2.75),
        bias=np.array([0.1,-0.2]),
        # bias=np.array([-0.07,-0.35]),
        legend_loc="lower right",
    )

    ## ASYMMETRIC P
    p = zelu(torch.arange(-1000, 1001, 1, dtype=torch.float32))
    p = torch.stack([p, -p]).T
    plot_means(
        path="/home/brothag1/code/stan/pooling/analysis/gem-jagged-asym-M_f.png",
        p=p,
        mu=mu,
        seed=11,
        M_f=True,
        xlim=(-2.75,3.25),
        ylim=(-2.75,2.75),
        bias=np.array([-0.1,0.2]),
        legend_loc="lower left"
    )

    ## TANH P
    p = zelu(torch.arange(-1000, 1001, 1, dtype=torch.float32))
    p = torch.stack([p, p]).T
    p[:,1] = zelu(np.round(np.tanh(0.005*p[:,1]) * 1000))
    plot_means(
        path="/home/brothag1/code/stan/pooling/analysis/gem-jagged-tanh-M_p.png",
        p=p,
        mu=mu,
        seed=11,
        M_f=False,
        xlim=(-2.75,3.25),
        ylim=(-2.75,2.75),
        bias=np.array([0.0,-0.3]),
        legend_loc="lower right"
    )
