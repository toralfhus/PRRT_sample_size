from utils import *

import warnings
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.utils import resample


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
    # If logfit=True the power-samplesize relationship is fitted to the logarithmic function pw = log(n),
    #   which may be used for extrapolating a coarse estimate for the needed population size to reach a power of interest

    df = pd.read_csv(path_load, index_col=0)

    p0 = df.iloc[0]
    df = df.drop(0)
    print("ANALYZING bootstrapped results from ", nm, df.shape)
    print("\tOriginal data:", p0.values)

    nvals = np.asarray(np.unique(df["n"].values), dtype=int)
    significance_rates = []

    for n in nvals:
        dfn = df[df["n"] == n].drop("n", axis=1)
        # print(dfn)
        perc_sign = dfn.apply(lambda r: stat_sign(*r), axis=1).sum() / len(dfn)
        print(n, dfn.shape, perc_sign)
        significance_rates.append(perc_sign)

    if fit:
        a, b = np.polyfit(np.log(nvals), significance_rates, deg=1)
        print(f"\tFitted logarithmic function: y={a:.2f} log(x) + {b:.2f}")
        fitfunc = lambda n: a * np.log(n) + b
        n_req = np.exp((desired_power - b) / a)

        if n_req < 1 or n_req > 1e4:
            print(f"\tbad estimate: n_req = {n_req:.3e} -> dropping log-fit")
            fit = False
            nvals_fit = None
        else:
            n_req = int(np.ceil(n_req))
            print(n_req)
            nvals_fit = list(range(min(nvals), n_req + 1))

    if plot:
        if len(plot) == 2:
            fig, ax = plot
        else:
            fig, ax = plt.subplots()

        print("\tplotting...")
        ax.plot(nvals, significance_rates, "o", label="Bootstrapped power")

        if fit:
            ax.plot(nvals_fit, fitfunc(nvals_fit), "--", label="Fitted log")

            # Plot horizontal line at desired power
            ax.axhline(y=desired_power, color='red', linestyle='--', label=f'Power = {desired_power}')

            # Plot vertical line at required n
            ax.axvline(x=n_req, color='green', linestyle='--', label=f'n ≈ {n_req}')

        p0 = {k: float(f"{v:.2g}") for k,v in p0.to_dict().items()}
        ax.set_title(f"{nm}\n{p0}")
        ax.grid()
        ax.set_xlabel("Sample size $n$")
        ax.set_ylabel("Rate of significance")
        ax.legend()

    return p0, nvals, significance_rates

