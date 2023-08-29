import numpy as np
import matplotlib.pyplot as plt


def format_log_axis(axis):
    axis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(
            lambda x, p: (
                "${:g}$".format(x)
                if (x <= 100) and (x >= 0.01)
                else "$10^{{{:g}}}$".format(np.log10(x))
            )
        )
    )


def mpl_settings():
    import matplotlib.pyplot as plt

    # plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["text.usetex"] = True
    # plt.rcParams["text.latex.preamble"] = r"\usepackage{txfonts}"
    plt.rcParams["font.size"] = 8.0
    plt.rcParams["legend.fancybox"] = False


#     plt.interactive(0)

source_name_labels = dict(
    RXJ1713="RX J1713.7$-$3946",
    VelaX="Vela X",
    GC="Galactic Centre",
    HESSJ1908="eHWC J1907+063",
    Westerlund1="Westerlund 1",
)


def feldman_cousins_errors(n_obs):
    """treat observation as counting experiment with zero background and lookup 68% Feldman-Cousins intervals."""
    n_obs = np.asarray(n_obs)
    if (not (n_obs % 1 == 0).all()) or (n_obs < 0).any():
        raise ValueError("Input must consist of positive integer values!")

    lookup = {
        0: (0.00, 1.29),
        1: (0.63, 1.75),
        2: (1.26, 2.25),
        3: (1.90, 2.30),
        4: (1.66, 2.78),
        5: (2.25, 2.81),
        6: (2.18, 3.28),
        7: (2.75, 3.30),
        8: (2.70, 3.32),
        9: (2.67, 3.79),
        10: (3.22, 3.81),
        11: (3.19, 3.82),
        12: (3.17, 4.29),
        13: (3.72, 4.30),
        14: (3.70, 4.32),
        15: (3.68, 4.32),
        16: (3.67, 4.80),
        17: (4.21, 4.81),
        18: (4.19, 4.82),
        19: (4.18, 4.82),
        20: (4.17, 5.30),
    }

    return np.array(list(map(lambda i: lookup.get(i, (np.sqrt(i), np.sqrt(i))), n_obs)))


# define a chi^2 function to calculate the best-fit chi^2 for fit models
def chi2(func, x, y, y_err):
    chi2 = (func(x) - y) ** 2 / y_err**2
    return chi2.sum()
