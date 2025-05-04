from utils import *

import warnings
from scipy.stats import pearsonr, ConstantInputWarning
from scipy.optimize import curve_fit
from sklearn.utils import resample
from sklearn.metrics import r2_score


def weibull_cdf(x, lb, k):
    return 1 - np.exp(-(x / lb) ** k)


def compute_bootstrapped_params(x_orig, y_orig, stat_names, stat_func, path_save=None, n_min=3, n_rep=100, nm=None):
    # Calculate statistical parameters (e.g. rho / p) using statfunc (pearson correlation)
    # for n_rep bootstrapped subsampled data of size n
    # from n_min to original size

    # index 0 is original (replicated) statistical parameters
    import warnings
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    n_orig = len(x_orig)
    print(f"\nCALCULATING {n_rep} bootstrap params from n={n_min} to n={n_orig}")
    stats_orig = stat_func(x_orig, y_orig)
    print(nm)
    print(stats_orig)

    df_boot = pd.DataFrame(columns=["n", *stat_names])
    df_boot.loc[0, ["n", *stat_names]] = [n_orig, *stats_orig]
    print(df_boot)

    j = 0
    print("N =", end=" ")
    for n in range(n_min, n_orig + 1):
        print(n, end=" ")

        for r in range(n_rep):
            j += 1
            xr, yr = resample(x_orig, y_orig, replace=True, n_samples=n)

            stats = stat_func(xr, yr)
            df_boot.loc[j] = [n, *stats]

    print()

    df_boot.to_csv(path_save)
    print("\tSAVED ", path_save)

    pass


def analyze_bootstrapped_params(path_load, stat_sign, nm=None, plot=None, fit=True, desired_power=0.9):
    # Using bootstrapped datasets of variable size to evaluate the type II error rate (power) in a monte-carlo setting
    # If logfit=True the power-samplesize relationship is fitted to
    # old: the logarithmic function pw = log(n)
    # new: weibull CDF (can approach both log or sigmoid shapes)
    # which may be used for extrapolating a coarse estimate for the needed population size to reach a power of interest

    df = pd.read_csv(path_load, index_col=0)

    p0 = df.iloc[0]
    df = df.drop(0)
    print("ANALYZING bootstrapped results from ", nm, df.shape)
    print("\tOriginal data:", p0.values)

    nvals = np.asarray(np.unique(df["n"].values), dtype=int)
    n_orig = max(nvals)
    significance_rates = []

    if df.columns.values[1] == "r":
        r = p0[1]
        p = p0[2]
        print(f"\toriginal r={r:.2f}, r2={r**2:.2f} (p={p:.3e})")

        # calculate 95% for pearson correlation r
        zhat = 0.5 * np.log((1 + r) / (1 - r))  # FIsher z-transform
        z95 = 1.96
        se = 1 / np.sqrt(n_orig - 3)
        z_low = zhat - z95 * se
        z_up = zhat + z95 * se

        r_ci = lambda z: (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
        r_low = r_ci(z_low)
        r_up = r_ci(z_up)
        print(f"\t95CI for R: ({r_low:.2f}, {r_up:.2f})")

        # t = r * np.sqrt((n_orig - 2) / (1 - r**2))
        # se = np.sqrt((1 - r**2) / (n_orig-2))


    print()
    for n in nvals:
        dfn = df[df["n"] == n].drop("n", axis=1)
        # print(dfn)
        perc_sign = dfn.apply(lambda r: stat_sign(*r), axis=1).sum() / len(dfn)
        # print(n, dfn.shape, perc_sign)
        print(".", end="")
        significance_rates.append(perc_sign)
    print()

    if fit:
        # a, b = np.polyfit(np.log(nvals), significance_rates, deg=1)
        # print(f"\tFitted logarithmic function: y={a:.2f} log(x) + {b:.2f}")
        # fitfunc = lambda n: a * np.log(n) + b
        # n_req = np.exp((desired_power - b) / a)
        init = [50, 2]
        bounds = [[1e-3, 1e-3], [1e3, 10]]
        (lb, k), _ = curve_fit(weibull_cdf, nvals, significance_rates, p0=init, bounds=bounds)
        fitfunc = lambda x: weibull_cdf(x, lb, k)
        print(f"\tFitted Weibull: lambda={lb}, k={k}")
        r2_fit = r2_score(significance_rates, weibull_cdf(nvals, lb, k))
        print(f"\t-> R2 for Weibull R2={r2_fit:.3f}")

        pw_orig = weibull_cdf(n_orig, lb, k)
        print(f"\t-> estimated power for original N={n_orig}: {pw_orig:.3f}")

        n_req = lb * (-np.log(1 - desired_power)) ** (1/k)
        print(n_req)

        if n_req < 1 or n_req > 1e4:
            print(f"\tbad estimate: n_req = {n_req:.3e} -> dropping log-fit")
            fit = False
            nvals_fit = None
        else:
            n_req = int(np.ceil(n_req))
            print(f"\t-> N to reach pwr={desired_power} -> {n_req}")
            if n_req > max(nvals):
                nvals_fit = list(range(min(nvals), n_req + 1))
            else:
                nvals_fit = list(range(min(nvals), max(nvals) + 1))

    if plot:
        if len(plot) == 2:
            fig, ax = plot
        else:
            fig, ax = plt.subplots()

        print("\tplotting...")
        ax.plot(nvals, significance_rates, "o", label="Bootstrapped power")

        if fit:
            ax.plot(nvals_fit, fitfunc(nvals_fit), "--", label=f"CWD fit (R2={r2_fit:.3f})")

            # Plot horizontal line at desired power
            ax.axhline(y=desired_power, color='red', linestyle='--', label=f'Power = {desired_power}')

            # Plot vertical line at required n
            ax.axvline(x=n_req, color='green', linestyle='--', label=f'n ≈ {n_req}')

        p0 = {k: float(f"{v:.2g}") for k,v in p0.to_dict().items()}
        ax.set_title(f"{nm}\n{p0}")
        ax.grid()
        ax.set_xlabel("Sample size $n$")
        # ax.set_ylabel("Rate of significance")
        ax.set_ylabel("Estimated power (1 - type II error rate)")
        ax.legend()

    return p0, nvals, significance_rates


def analyze_bootsrapped_negative_results(path_load, nm=None, plot=None, desired_power=0.9):
    df = pd.read_csv(path_load, index_col=0)

    p0 = df.iloc[0]
    df = df.drop(0)
    print("ANALYZING bootstrapped results from ", nm, df.shape)
    n_orig, r, p = p0.values
    print(f"\tOriginal data: n={n_orig}, r={r:.3f}, p={p:.3f}")
    nvals = np.asarray(np.unique(df["n"].values), dtype=int)

    delta_r = 0.5   # effect size to determine equivalence power for
    alpha = 0.05

    df = df[df["n"] > 3].reset_index(drop=True)

    df["z"] = np.arctanh(df["r"])
    df["se"] = 1.0 / np.sqrt(df["n"] - 3)


    z_lo = np.arctanh(-0.5)
    z_hi = np.arctanh(+0.5)

    from scipy.stats import norm

    df["p_low"] = 1 - norm.cdf((df["z"] - z_lo) / df["se"])
    df["p_high"] = norm.cdf((df["z"] - z_hi) / df["se"])

    df['equiv_success'] = (df['p_low'] < alpha) & (df['p_high'] < alpha)
    # print(df)

    power_equiv = df.groupby("n")["equiv_success"].mean()
    # print(power_equiv)
    nvals, power_equiv = power_equiv.index.values, power_equiv.values

    (lb, k), _ = curve_fit(weibull_cdf, nvals, power_equiv, p0=[50, 2], bounds=[[1e-3, 1e-3], [1e3, 10]])
    fitfunc = lambda x: weibull_cdf(x, lb, k)

    print(f"\tFitted Weibull: lambda={lb}, k={k}")
    r2_fit = r2_score(power_equiv, weibull_cdf(nvals, lb, k))
    print(f"\t-> R2 for Weibull R2={r2_fit:.3f}")
    pw_orig = weibull_cdf(n_orig, lb, k)
    print(f"\t-> estimated power for original N={n_orig}: {pw_orig:.3f}")
    n_pow_fit = lb * (-np.log(1 - desired_power)) ** (1 / k)
    n_pow_fit = int(np.ceil(n_pow_fit))
    print("\tweibull n:", n_pow_fit)

    n_pow_analytic = ((1.645 + 1.282) / np.arctanh(delta_r))**2 + 3
    n_pow_analytic = int(np.ceil(n_pow_analytic))
    print("\tanalytic n:", n_pow_analytic)

    nvals_fit = list(range(int(min(nvals)), max([max(nvals), n_pow_fit]) + 1))

    fig, ax = plot
    ax.plot(nvals, (1-power_equiv), "o")

    ax.plot(nvals_fit, (1-fitfunc(nvals_fit)), "--", label=f"CWD fit (R2={r2_fit:.3f})")
    # Plot horizontal line at desired power
    ax.axhline(y=1-desired_power, color='red', linestyle='--', label=f'Equivalence power = {desired_power}')

    # Plot vertical line at required n
    ax.axvline(x=n_pow_fit, color='green', linestyle='--', label=f'n ≈ {n_pow_fit}')

    p0 = {k: float(f"{v:.2g}") for k, v in p0.to_dict().items()}
    ax.set_title(f"{nm}\n{p0}")
    ax.grid()
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Estimated type II error rate (1 - equivalence power)")
    ax.legend()

    # plt.show()

    return p0, nvals, power_equiv

