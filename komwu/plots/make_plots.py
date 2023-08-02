import json
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("pgf")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": False,    # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{times}",
    ])
})

GAMES = [
    "K312",
    "K45",
    "L3133",
    "L4133",
]

def add_line(ax, game_stem, algo, **kwargs):
    data = json.load(open(f'../logs/{game_stem}_{algo}.json'))

    xs = []
    ys = []
    for dp in data:
        xs += [dp["iteration"]]
        ys += [max(dp["regrets"])]
    ax.loglog(xs, ys, **kwargs)

    m = None
    if "eta1.0" in algo:
        m = 'o'
    elif 'eta10.0' in algo:
        m = 'v'
    elif 'eta5.0' in algo:
        m = 's'
    elif 'eta0.1' in algo:
        m = '>'

    if m is not None:
        col = kwargs["c"]
        pos = int(len(xs) * .05)
        return ax.loglog(xs[pos], ys[pos], m, c=col, fillstyle='none')[0]


def make_komwu_vs_domwu():
    fig, axs = plt.subplots(2,2,figsize=(4.5,4),sharex=True)

    def add_game(game, ax, title=None):
        if title is None:
            title = game

        z1 = add_line(ax, game, "komwu_eta1.0", c="#377eb8", ls="--", label="KOWMU")
        z10 = add_line(ax, game, "komwu_eta10.0", c="#377eb8", ls="--")
        z5 = add_line(ax, game, "komwu_eta5.0",  c="#377eb8", ls="--")
        z01 = add_line(ax, game, "komwu_eta0.1", c="#377eb8", ls="--")

        add_line(ax, game, "domwu_eta1.0", c="#ff7f00", ls="dotted", label="DOWMU")
        add_line(ax, game, "domwu_eta10.0", c="#ff7f00", ls="dotted")
        add_line(ax, game, "domwu_eta5.0",  c="#ff7f00", ls="dotted")
        add_line(ax, game, "domwu_eta0.1", c="#ff7f00", ls="dotted")


        ax.grid()
        ax.set_title(title)

        return [z01, z1, z5, z10]


    add_game(GAMES[0], axs[0,0], title="3-player Kuhn poker")
    add_game(GAMES[1], axs[0,1], title="4-player Kuhn poker")
    add_game(GAMES[2], axs[1,0], title="3-player Leduc poker")
    [z01, z1, z5, z10] = add_game(GAMES[3], axs[1,1], title="4-player Leduc poker")
    fig.tight_layout()


    axs[1,1].legend([z01, z1, z5, z10], ["$\eta=0.1$", "$\eta=1$", "$\eta=5$", "$\eta=10$"], ncol=4, handletextpad=0.0, columnspacing=.5, loc=3, bbox_to_anchor=(-.5, -.55), fontsize='small')
    axs[1,0].legend(fontsize='small', ncol=3, loc=4, handletextpad=0.5, columnspacing=1.1, bbox_to_anchor=(0.77, -.55))
    axs[0,0].set_ylabel("Max. individual regret")
    axs[1,0].set_ylabel("Max. individual regret")
    axs[1,0].set_xlabel("Iteration")
    axs[1,1].set_xlabel("Iteration")

    axs[1,0].set_ylim(1e0,2e3)

    fig.savefig(f'komwu_vs_domwu.pdf', bbox_inches='tight')



def make_komwu_vs_cfr():
    fig, axs = plt.subplots(2, 2,figsize=(6,4),sharex=True)

    def add_game(game, ax, title=None):
        if title is None:
            title = game

        add_line(ax, game, "cfr", c="#ff7f00", label="CFR")
        add_line(ax, game, "cfr_rmp", c="#f781bf", label="CFR(RM+)", ls="-.")
        z1 = add_line(ax, game, "komwu_eta1.0", c="#377eb8", ls="--", label="KOWMU")
        z10 = add_line(ax, game, "komwu_eta10.0", c="#377eb8", ls="--")
        z5 = add_line(ax, game, "komwu_eta5.0",  c="#377eb8", ls="--")
        z01 = add_line(ax, game, "komwu_eta0.1", c="#377eb8", ls="--")

        ax.grid()
        ax.set_title(title)

        return [z01, z1, z5, z10]

    add_game(GAMES[0], axs[0,0], title="3-player Kuhn poker")
    add_game(GAMES[1], axs[0,1], title="4-player Kuhn poker")
    add_game(GAMES[2], axs[1,0], title="3-player Leduc poker")
    [z01, z1, z5, z10] = add_game(GAMES[3], axs[1,1], title="4-player Leduc poker")
    fig.tight_layout()

    axs[1,1].legend([z01, z1, z5, z10], ["$\eta=0.1$", "$\eta=1$", "$\eta=5$", "$\eta=10$"], ncol=4, handletextpad=0.0, columnspacing=.5, loc=3, bbox_to_anchor=(-.05, -.55), fontsize='small')
    axs[1,0].legend(fontsize='small', ncol=3, loc=4, handletextpad=0.5, columnspacing=1.1, bbox_to_anchor=(1.0, -.55))
    axs[0,0].set_ylabel("Max. individual regret")
    axs[1,0].set_ylabel("Max. individual regret")
    axs[1,0].set_xlabel("Iteration")
    axs[1,1].set_xlabel("Iteration")

    axs[1,0].set_ylim(1e0,2e3)
    axs[1,1].set_ylim(1e0,2e3)


    fig.savefig(f'komwu_vs_cfr.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # make_komwu_vs_domwu()
    make_komwu_vs_cfr()