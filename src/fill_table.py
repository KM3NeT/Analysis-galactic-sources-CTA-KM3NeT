import numpy as np

from scipy.integrate import quad
from scipy.optimize import curve_fit, newton, root
from scipy.interpolate import interp1d

from glob import glob
from pathlib import Path
import os
import sys

import pandas as pd
import re

from configure_analysis import AnalysisConfig

analysisconfig = AnalysisConfig()

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(Path(file).name.split("_")[-1])
    if not match:
        return np.inf
    return int(match.groups()[0])


def f_int(x, deltaTS, shift):
    """Returns the PDF from the likelihood ratios for integration: L1/L2 = exp(-deltaTS/2)
    x: array of values of f between 0 and 1
    deltaTS: callable interpolated function of deltaTS values
    shift: shift added to the Gaussian PDF. Used for root finding
    """
    return np.exp(-deltaTS(x) / 2) - shift


def get_x(y, deltaTS):
    """Returns the values of f at which the PDF has the value y.
    Used to get the integration limits.
    y: float
    deltaTS: callable interpolated function of deltaTS values
    """
    n = 1e3
    x_test = np.linspace(0, 1, int(n + 1))
    y_test = f_int(x_test, deltaTS, y)

    # find the peak of the function (center of the Gauss)
    x_max = y_test.argmax()
    # case there are no roots
    if y_test.max() < 0:
        # y is too large so all values are negative --> integration range = 0
        return np.array([x_max / n, x_max / n])
    elif y_test.min() > 0:
        # y is negative to all values are > 0 --> integration range [0,1]
        return np.array([0, 1])

    # get a value of x right to the peak where f(x) is close to y
    x_start2 = (np.abs(y_test[x_max:]).argmin() + x_max) / n

    if x_start2 == 1:
        x2 = 1
    else:
        x2 = root(f_int, x0=x_start2, args=(deltaTS, y), tol=1e-8).x[0]
    if x2 > 1 or x2 < 0 or np.abs(x2 - x_start2) > 0.2:
        x2 = x_start2  # need this because root fails sometimes
    # get a value of x left to the peak where f(x) is close to y
    x_start1 = np.abs(y_test[: x_max + 1]).argmin() / n  # for lep

    if x_start1 == 0:
        x1 = 0
    else:
        x1 = root(f_int, x0=x_start1, args=(deltaTS, y), tol=1e-8).x[0]
    if x1 > 1 or x1 < 0 or np.abs(x1 - x_start1) > 0.2:
        x1 = x_start1  # need this because root fails sometimes
    # check if something still went wrong
    #     if x1==x2 or x1<0 or x1>1 or x2<0 or x2>1:
    #         raise RuntimeError('x1 or x2 are not correct')

    return x1, x2


def get_integral_for_y(y, q, deltaTS, int_total):
    """
    Returns the shifted integral value for a given y.
    y: float
    q: float, the quantile of the total integral one wants to obtain
    deltaTS: callable interpolated function of deltaTS values
    int_total: float, the total integral from [0,1] of f_int()
    """
    x1, x2 = get_x(y, deltaTS)
    int_x = quad(f_int, x1, x2, args=(deltaTS, 0))[0]
    return int_x - int_total * q


def get_integral_for_ul(x, q, deltaTS, int_total, limit="upper"):
    if limit == "upper":
        x1 = 0
        int_x = quad(f_int, x1, x, args=(deltaTS, 0))[0]
    elif limit == "lower":
        x2 = 1
        int_x = quad(f_int, x, x2, args=(deltaTS, 0))[0]
    else:
        raise ValueError("limit must be 'lower' or 'upper'")
    return int_x - int_total * q


def fill_table(
    source_names,
    q,
    filename_table=None,
    data_path=analysisconfig.get_file("likelihood_analysis"),
    n_pex=100,
):
    """
    source_names: list of str, source names for which the table should be filled/updated
    q: float, quantile of the error

    The function fills/updates the results_q table in the main directory
    """
    if filename_table is None:
        filename_table = analysisconfig.get_file(
            "likelihood_analysis/results_q" + str(int(q * 100)) + ".csv"
        )

    result_df = pd.DataFrame(
        columns=[
            "name",
            "f_input",
            "f_avg",
            "f_errn",
            "f_errp",
            "f_std",
            "errn_std",
            "errp_std",
            "quantile",
        ]
    )
    result_df.set_index("name", inplace=True)
    for source_name in source_names:
        for f_input in [0, 1]:
            df = pd.read_csv(
                analysisconfig.get_file(
                    "likelihood_analysis/" + source_name + "_" + str(f_input) + ".0.csv"
                )
            )
            gdf = df.groupby(["seed", "case"])
            stat_tot = gdf["stat_total"].apply(lambda v: np.array(v)).unstack()
            TS3 = np.vstack(stat_tot[3])

            int_PD = gdf["int_PD"].apply(lambda v: np.array(v)).unstack()
            int_PD3 = np.vstack(int_PD[3])

            int_IC = gdf["int_IC"].apply(lambda v: np.array(v)).unstack()

            int_IC3 = np.vstack(int_IC[3])

            int_PD3_avg = np.mean(int_PD3, axis=0)
            int_IC3_avg = np.mean(int_IC3, axis=0)

            f = np.array(int_PD3) / (np.array(int_PD3) + np.array(int_IC3))
            ts_vals = TS3
            analysis = "comb"
            f_avg = []
            f_errn = []
            f_errp = []

            # Get limits for each individual scan
            limits = []
            f_meas = []
            for i, ts in enumerate(ts_vals):
                deltaTS = interp1d(
                    f[i],
                    ts,
                    kind="cubic",
                    assume_sorted=True,
                    bounds_error=False,
                    fill_value=np.inf,
                )
                int_total = quad(f_int, 0, 1, args=(deltaTS, 0))[0]
                # evaluate on fine grid
                n = 1e3
                x_test = np.linspace(f[i][0], f[i][-1], int(n + 1))
                y_test = f_int(x_test, deltaTS, 0)
                f_best = x_test[y_test.argmax()]
                f_meas.append(f_best)  # value of f with the most probability
                if q >= 0.9:
                    y_start = y_test.max() * 0.26 + y_test.min() * (
                        1 - 0.26
                    )  # y90 is expected at 26% of max for half-sided Gaussian
                else:
                    y_start = (y_test.max() + y_test.min()) / 2
                y90 = root(
                    get_integral_for_y,
                    x0=y_start,
                    args=(q, deltaTS, int_total),
                    tol=1e-8,
                ).x[0]
                x90 = get_x(y90, deltaTS)
                limits.append(x90)

            if f_input == 0:
                quant = q
            elif f_input == 1:
                quant = 1 - q

            f_meas = np.array(f_meas)  # [mask]
            limits = np.array(limits)  # [mask]

            f_avg = np.mean(f_meas)
            f_quantile = np.quantile(f_meas, quant)
            f_std = np.std(f_meas)
            errs_avg = np.mean(limits, axis=0)
            errs_std = np.std(limits, axis=0)
            f_errn = f_avg - errs_avg[0]
            f_errp = errs_avg[1] - f_avg
            sqrt_n = np.sqrt(n_pex)
            data_dict = dict(
                f_input=f_input,
                f_avg=f_avg,
                f_errn=f_errn,
                f_errp=f_errp,
                f_std=f_std / sqrt_n,
                errn_std=errs_std[0] / sqrt_n,
                errp_std=errs_std[1] / sqrt_n,
                quantile=f_quantile,
            )
            result_series = pd.Series(
                data=data_dict, name=f"{source_name}_{f_input:.2f}_{analysis}"
            )

            result_df.loc[result_series.name] = result_series
            print(f"Finished {source_name} - {f_input} - {analysis}")

    result_df.sort_index(inplace=True)
    result_df.to_csv(filename_table)
