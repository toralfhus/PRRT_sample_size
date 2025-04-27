import os
import numpy as np
from pyparsing import opAssoc
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression, LogisticRegression
from matplotlib import pyplot as plt
import sys
import pandas as pd
from sympy.stats import Logistic

def get_logistic_prediction_scores(ytrue, phat, p_thresh=0.5):
    from sklearn.metrics import confusion_matrix, brier_score_loss, roc_auc_score, accuracy_score


    yhat = phat.copy()
    yhat[yhat >= p_thresh] = 1
    yhat[yhat < p_thresh] = 0
    yhat = yhat.astype(int)

    cm = confusion_matrix(ytrue, yhat)
    # print(cm)

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    # FNR / FPR instead of TPR / TNR ?
    PPV = TP / (TP + FP)    # = precision
    TPR = TP / (TP + FN)    # = power, recall, sensitivity  -> 1 - type II error rate
    TNR = TN / (FP + TN)    # = specificity -> 1 - type I error rate
    NPV = TN / (FN + TN)    # = 1 - false discovery rate

    BS = brier_score_loss(ytrue, phat)
    AUC = roc_auc_score(ytrue, phat)
    ACC = accuracy_score(ytrue, yhat)

    print(f"\tPPV={PPV:.3f}, TPR={TPR:.3f}, TNR={TNR:.3f}, NPV={NPV:.3f}, AUC={AUC:.3f}, BS={BS:.3f}, ACC={ACC:.3f}")


    return PPV, TPR, TNR, NPV, AUC, BS, ACC


class LogitRegression(LinearRegression):
    # super().fit_intercept = True
    def fit(self, x, p):
        p = np.asarray(p)
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)


def estimate_lognorm_params(median, pc_low, pc_high, z=2.33):
    # z = 1.645   # 5%
    # z = 2.33    # standard normal one-tailed critical value @ 1%
    # z = 2.6
    # z = 3.09    # 0.1%
    mu = np.log(median)
    sigma = (np.log(pc_high) - np.log(pc_low)) / (2 * z)

    return mu, sigma


def search_lognorm_params_to_logistic_fit_old(x_true, p_true, z0=None, max_iter=10, N_fit=100):
    # x_true, p_true logistic curv to estimate
    # by fitting two lognorm distributions using a range of data (median, pc low / high)
    # -> optimize z-value in parameter conversion function (estimate_lognorm_params)
    from sklearn.metrics import mean_squared_error

    noresp_params = [72, 8, 136]
    resp_params = [126, 21, 228]

    # z0 = 1.5
    # mu0, s0 = estimate_lognorm_params(*noresp_params, z=z0)
    # mu1, s1 = estimate_lognorm_params(*resp_params, z=z0)
    mu0, s0 = 4.0, 0.6
    mu1, s1 = 4.8, 0.3
    print(mu0, s0)
    print(mu1, s1)
    print(N_fit)

    x_0 = np.random.lognormal(mu0, s0, size=50)
    x_1 = np.random.lognormal(mu1, s1, size=250)

    # x_0 = np.random.normal(loc=72, scale=30, size=N_fit)
    # x_1 = np.random.normal(loc=126, scale=50, size=N_fit)

    # x_0 = np.linspace(8, 136, N_fit)
    # x_1 = np.linspace(21, 228, N_fit)

    x = np.array([*x_0, *x_1]).reshape(-1, 1)
    yhat = np.array([*np.repeat(0, len(x_0)), *np.repeat(1, len(x_1))]).reshape(-1, 1)

    lr = LogisticRegression(penalty=None, fit_intercept=True)
    lr.fit(x, yhat)
    print(lr.intercept_, lr.coef_)
    phat = lr.predict_proba(x)[:, -1]

    mse = mean_squared_error(p_true, phat)
    print(f"MSE={mse:.3g}")

    plt.plot(x, yhat, "o")
    plt.plot(x, phat, "x", label="fit")
    plt.plot(x_true, p_true, ":", label="true")
    plt.show()


    pass


def search_lognorm_params_to_logistic_fit(xvals, p_true, init_params=None):
    from scipy.optimize import minimize, differential_evolution
    from sklearn.metrics import mean_squared_error

    def objective_function(params):

        mu0, s0, mu1, s1 = params
        x0 = np.random.lognormal(mu0, s0, size=n_samples)
        x1 = np.random.lognormal(mu1, s1, size=n_samples)
        x = np.concatenate([x0, x1]).reshape(-1, 1)
        y = np.array([0] * n_samples + [1] * n_samples)

        m = LogisticRegression(penalty=None, fit_intercept=True)
        m.fit(x, y)

        # phat = m.predict_proba(x)[:, -1]
        phat = m.predict_proba(xvals)[:, -1]
        # score = np.sum((phat - p_true)**2) / len(phat)
        # score = np.average((phat - p_true)**2)
        score = mean_squared_error(p_true, phat)

        print(params, f"\t-> mse={score:.3e}")
        # sys.exit()
        return score

    n_samples = 10000

    print(f"INITIAL PARAMS = ", init_params)
    se0 = objective_function(init_params)
    print("SE0=", se0)

    # bounds = [(1, 5), (0.1, 1), (1, 5), (0.1, 1)]
    bounds = [(0.1, 6), (0.1, 1), (0.1, 6), (0.1, 1)]

    res = minimize(objective_function, init_params, method='Powell', bounds=bounds,
                   options={"maxiter":10}, tol=1e-6)

    # res = differential_evolution(objective_function, bounds,
    #                                 strategy='best1bin', maxiter=100000, popsize=15, tol=0.01, mutation=(0.5, 1.5),
    #                                 recombination=0.7)
    print(res)
    print(res.x)
    se = objective_function(res.x)
    print(f"SE={se:.3f} (SE0={se0:.3f})")

    return res.x


def analytic_sample_size_estimate(mean_0, mean_1, sd0, sd1, alpha=0.05, beta=0.1):
    from scipy.stats import norm

    epsilon2 = (mean_1 - mean_0) ** 2
    z_alpha = norm.ppf(q=1 - alpha, loc=0, scale=1)
    z_beta = norm.ppf(q=1 - beta, loc=0, scale=1)
    n_min_1 = np.ceil(2 * (z_alpha * sd0 + z_beta * sd1) ** 2 / epsilon2)
    n_min_2 = np.ceil((z_alpha + z_beta) ** 2 * (sd0 ** 2 + sd1 ** 2) / epsilon2)
    # print(f"means = {mean_0:.2f} / {mean_1:.2f} (squared diff = {epsilon2:.2f})\nstds = {sd0:.2f} / {sd1:.2f}")
    # print(f"z_alpha from H0 = {z_alpha:.2f}, z_beta from Ha = {z_beta:.2f}")
    print(f"Minimum n_1 (analytic) given alpha={alpha}, beta={beta} = {n_min_1:.0f} / {n_min_2:.0f}")
    pass


def monte_carlo_sample_size_calculate(sample_func, params_0, params_1, Nmax=100, num_repeat=1000, savename=None, shuffle=False):
    from scipy.stats import mannwhitneyu, ttest_ind, chisquare

    # from sklearn.feature_selection import chi2
    # if shuffle: create pseudo-distribution of phi0, phia by shuffling outcome vector y (permutation-distribution) -> H0 true

    if os.path.exists(savename):
        df_pvals = pd.read_csv(savename, index_col=0)
        # print(df_pvals)
        N0 = int(df_pvals["n"].max()) + 1
        i = int(df_pvals.index.max())
        # print(N0, i)
    else:
        df_pvals = pd.DataFrame(dtype=float)
        N0 = 5
        i = -1

    for N in range(N0, Nmax):
        for r in range(num_repeat):
            i += 1
            # s0, mu0 = params_0
            # s1, mu1 = params_1
            # x0 = sample_func(mu0, s0, size=N//2)
            # x1 = sample_func(mu1, s1, size=N-len(x0))

            x0 = sample_func(*params_0, size=N//2)
            x1 = sample_func(*params_1, size=N-len(x0))

            x = np.concatenate([x0, x1]).reshape(-1, 1)
            y = np.array([0] * len(x0) + [1] * len(x1)).reshape(-1, 1)

            if shuffle:
                np.random.shuffle(y)
                x0 = x[y == 0]
                x1 = x[y == 1]

            # TODO: add logistic regression (how?)
            mwu = mannwhitneyu(x0, x1)
            mwu_stat, mwu_p = float(mwu.statistic), float(mwu.pvalue)
            t_stat, t_p = ttest_ind(x0, x1, equal_var=False)

            mwu_p_log = mannwhitneyu(np.log(x0), np.log(x1)).pvalue
            _, t_p_log = ttest_ind(np.log(x0), np.log(x1), equal_var=False)

            # # lr = LogisticRegression(fit_intercept=True)
            # # lr.fit(x, y)
            # chi2_stat, chi2_p = chisquare(x0, x1)

            print(f"N={N}, r={r}", end="\t")
            print(f"T-test p={t_p:.2e}, MWU p={mwu_p:.2e}, T-log p={t_p_log:.2e}, MWU-log p={mwu_p_log:.2e}")

            import seaborn as sns
            # plt.hist(x.ravel(), color=[f"C{i}" for i in y.ravel()])
            # plt.hist(x.ravel())
            # print(x.ravel())
            # print(y.ravel())
            # print(chi2_p)
            # plt.show()
            # sys.exit()

            # df_pvals.loc[i, ["n", "t_p", "mwu_p"]] = [N, t_p, mwu_p]
            df_pvals.loc[i, ["n", "t_p", "mwu_p", "log_t_p", "log_mwu_p"]] = [N, t_p, mwu_p, t_p_log, mwu_p_log]
            # print(df_pvals)

        if (not N%10) and savename:
            df_pvals.to_csv(savename)
            print("saved to", savename)

    if savename:
        df_pvals.to_csv(savename)

    pass


def roc_analysis(y, phat):
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score

    auc = roc_auc_score(y, phat)
    fpr, tpr, ths = roc_curve(y, phat)
    print(y.shape, len(np.unique(phat)), len(ths))

    print(f"auc={auc:.3f}")

    idx_half = np.argmin([np.abs(ths - 0.5)])
    yhat_half = phat.copy()
    yhat_half[yhat_half < 0.5] = 0
    yhat_half[yhat_half >= 0.5] = 1
    acc_half = accuracy_score(y, yhat_half)
    prec_half, rec_half = precision_score(y, yhat_half), recall_score(y, yhat_half)
    print(f"acc using ths=0.5 -> {acc_half:.2f}, prec={prec_half:.2f}, rec={rec_half:.2f}")
    # print(idx_half, ths[idx_half])
    # sys.exit()

    youden_index = tpr - fpr
    idx_yo = np.argmax(youden_index)
    ths_yo = ths[idx_yo]
    yhat_yo = phat.copy()
    yhat_yo[yhat_yo < ths_yo] = 0
    yhat_yo[yhat_yo >= ths_yo] = 1
    acc_yo = accuracy_score(y, yhat_yo)
    prec_yo, rec_yo = precision_score(y, yhat_yo), recall_score(y, yhat_yo)
    print(f"acc using ths={ths_yo:.2f} (youden) -> {acc_yo:.2f}, prec={prec_yo:.2f}, rec={rec_yo:.2f}")
    # print(f"acc using ths={ths_yo:.3f} (max youden) -> {acc_yo}")

    # d_eucl = np.array([fpr - 0, tpr - 1])
    d_eucl = np.array([fpr - 1, tpr - 0])
    d_eucl = np.array([f**2 + t**2 for f, t in d_eucl.T])
    idx_eucl = np.argmin(d_eucl)
    ths_eucl = ths[idx_eucl]
    # print(d_eucl.shape, idx_eucl, ths_eucl)
    yhat_eucl = phat.copy()
    yhat_eucl[yhat_eucl < ths_eucl] = 0
    yhat_eucl[yhat_eucl >= ths_eucl] = 1
    acc_eucl = accuracy_score(y, yhat_eucl)
    print(f"acc using ths={ths_eucl:.3f} (min eucl dist to 0, 1) -> {acc_eucl}")

    accvals = []
    precvals = []
    recvals = []
    for t in ths:
        yhat_t = phat.copy()
        yhat_t[yhat_t < t] = 0
        yhat_t[yhat_t >= t] = 1

        acc = accuracy_score(y, yhat_t)
        prec = precision_score(y, yhat_t)
        rec = recall_score(y, yhat_t)

        accvals.append(acc)
        precvals.append(prec)
        recvals.append(rec)

    idx_acc = np.argmax(accvals)
    ths_acc = ths[idx_acc]
    print(f"acc using ths={ths_acc:.3f} (max acc) -> {accvals[idx_acc]:.2f}")

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(fpr, tpr, "-")
    ax[0].plot(fpr[idx_half], tpr[idx_half], "o", label="ths=0.5")
    ax[0].plot(fpr[idx_yo], tpr[idx_yo], "o", label=f"ths={ths_yo:.2f}")
    ax[0].plot(fpr[idx_acc], tpr[idx_acc], "o", label=f"ths={ths_acc:.2f}")
    ax[0].set_xlabel("fpr")
    ax[0].set_ylabel("tpr")
    ax[0].set_title(f"auc={auc:.3f}")
    ax[0].legend()

    ax[1].plot(ths, youden_index, label="youden")
    ax[1].plot(ths, accvals, label="acc")
    ax[1].plot(ths, precvals, label="precicion")
    ax[1].plot(ths, recvals, label="recall")
    ylims = ax[1].get_ylim()
    ax[1].vlines(ths_yo, ymin=ylims[0], ymax=ylims[1], ls=":")
    ax[1].set_ylim(ylims)
    ax[1].set_xlabel("thresh")
    ax[1].legend()
    plt.show()

    pass


def repeated_bootstrap_samplesize_calculate(x, y, num_repeat=1000, stratify=True, shuffle=False, savename=None):
    from sklearn.utils import resample
    from scipy.stats import mannwhitneyu, ttest_ind, chisquare

    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)

    if os.path.exists(savename):
        df_pvals = pd.read_csv(savename, index_col=0)
        N0 = int(df_pvals["n"].max()) + 1
        i = int(df_pvals.index.max())
    else:
        df_pvals = pd.DataFrame(dtype=float)
        N0 = 5
        i = -1

    Nmax = len(x)   # Nmax > len(x) -> oversample using bootstrap?

    for N in range(N0, Nmax):
        for r in range(num_repeat):
            i += 1
            x_r, y_r = resample(x, y, replace=True, n_samples=N, stratify=y if stratify else None)
            # print(x_r.shape, y_r.shape, np.unique(y_r, return_counts=True))

            if len(np.unique(y_r).ravel()) == 1:
                pass
            elif 1 in np.unique(y_r, return_counts=True)[-1]:
                pass
            else:
                if shuffle:
                    np.random.shuffle(y_r)

                x_r0 = x_r[y_r == 0]
                x_r1 = x_r[y_r == 1]
                # print(len(x_r0), len(x_r1))

                p_mwu = mannwhitneyu(x_r0, x_r1).pvalue
                _, p_t = ttest_ind(x_r0, x_r1, equal_var=False)
                _, p_t_log = ttest_ind(np.log(x_r0), np.log(x_r1), equal_var=False)

                print(f"N={N}, r={r}", end="\t")
                print(f"T-test p={p_t:.2e}, MWU p={p_mwu:.2e}, T-log p={p_t_log:.2e}")

                df_pvals.loc[i, ["n", "t_p", "mwu_p", "log_t_p"]] = [N, p_t, p_mwu, p_t_log]


        if (not N%10) and savename:
            df_pvals.to_csv(savename)
            print("saved to", savename)

    if savename:
        df_pvals.to_csv(savename)
    pass


def monte_carlo_sample_size_analyze_typeII(savename, alpha=0.05, power=0.9, title=""):
    # power = 1 - type II error rate -> Ha true, fail to reject
    # must keep alpha constant to estimate power for varying sample sizes?

    ha_true = not("shuff" in savename)

    df = pd.read_csv(savename, index_col=0)
    print("Loaded p-vals:", df.shape)
    # print(df.columns.values)
    log = any(["log" in c for c in df.columns.values])

    # print(df)

    nvals = np.unique(df["n"].values)

    df_tpr = pd.DataFrame() # False positive rate if Ha is false
    df_tnr = pd.DataFrame() # false negative rate if Ha is true


    for n in nvals:
        pvals_t = df[df["n"] == n]["t_p"].values
        pvals_mwu = df[df["n"] == n]["mwu_p"].values
        # pvals_log_mwu = df[df["n"] == n]["log_mwu_p"].dropna().values

        tpr_t = len(pvals_t[pvals_t < alpha]) / len(pvals_t)
        tpr_mwu = len(pvals_mwu[pvals_mwu < alpha]) / len(pvals_mwu)

        if log:
            pvals_log_t = df[df["n"] == n]["log_t_p"].dropna().values
            tpr_log_t = len(pvals_log_t[pvals_log_t < alpha]) / len(pvals_log_t)
            df_tpr.loc[n, ["tpr_t", "tpr_mwu", "tpr_log_t"]] = [tpr_t, tpr_mwu, tpr_log_t]
        else:
            df_tpr.loc[n, ["tpr_t", "tpr_mwu"]] = [tpr_t, tpr_mwu]
        # df_tpr.loc[n, ["tpr_t", "tpr_mwu", "tpr_log_t", "tpr_log_mwu"]] = [tpr_t, tpr_mwu, tpr_log_t, tpr_log_mwu]

    print("Calculated TPR:", df_tpr.shape)

    nmin_t = np.min(df_tpr[df_tpr["tpr_t"] > power].index)
    nmin_mwu = np.min(df_tpr[df_tpr["tpr_mwu"] > power].index)
    nmin_log_t = np.min(df_tpr[df_tpr["tpr_log_t"] > power].index) if log else 0
    # nmin_log_mwu = np.min(df_tpr[df_tpr["tpr_log_mwu"] > power].index)

    ms = 3
    plt.plot(nvals, df_tpr["tpr_mwu"], "o", label=f"MWU (N_min={nmin_mwu})", alpha=0.7, ms=ms)
    plt.plot(nvals, df_tpr["tpr_log_t"], "o", label=f"log-T (N_min={nmin_log_t})", alpha=0.7, ms=ms) if log else 0
    plt.plot(nvals, df_tpr["tpr_t"], "o", label=f"T-test (N_min={nmin_t})", alpha=0.7, ms=ms)
    # plt.plot(nvals, df_tpr["tpr_log_mwu"], "o", label=f"log-MWU (N_min={nmin_log_mwu})", alpha=0.7, ms=ms)

    xlims = plt.xlim()
    ylims = plt.ylim()

    plt.hlines([power], xmin=xlims[0], xmax=xlims[1], zorder=-1, color="black", ls=":")
    # plt.vlines([nmin_t, nmin_mwu], ymin=ylims[0], ymax=ylims[1], zorder=-1, color="black", ls=":")

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.legend()
    plt.xlabel("Number of observations (lesion doses)")
    ylab = "TPR / power ($1-\\beta$)" if ha_true else "FPR / type I error rate"
    plt.ylabel(ylab)
    title += "\n" if title else ""
    title += f"{ha_true} positive rate (Ha {ha_true}) given $\\alpha$={alpha:.2g}, power={power}"
    plt.title(title)
    plt.show()

    pass


def fit_and_draw_lognorms(x0, x1, Ndraw=1000, plot=False, evaluate=True, return_params=False, verbose=True):
    from scipy.stats import lognorm
    print(f"FITTING log-norm distributions with {len(x0)} and {len(x1)} observations") if verbose else 0
    p0 = lognorm.fit(x0)
    p1 = lognorm.fit(x1)
    print("\tparams0=", p0) if verbose else 0
    print("\tparams1=", p1) if verbose else 0
    if return_params:
        return p0, p1

    if Ndraw is None:
        x_ln0 = lognorm.rvs(*p0, size=len(x0))
        x_ln1 = lognorm.rvs(*p1, size=len(x1))
    else:
        x_ln0 = lognorm.rvs(*p0, size=Ndraw//2)
        x_ln1 = lognorm.rvs(*p1, size=Ndraw - len(x_ln0))
        # print(Ndraw)

    if evaluate:
        from scipy.stats import mannwhitneyu, ttest_ind
        u_orig, pu_orig = mannwhitneyu(x0, x1)
        u_ln, pu_ln = mannwhitneyu(x_ln0, x_ln1)
        print(f"\tMWU: \t\tp_orig={pu_orig:.3e}, p_drawn={pu_ln:.3e}")
        _, pt_orig = ttest_ind(x0, x1, equal_var=False)
        _, pt_ln = ttest_ind(x_ln0, x_ln1, equal_var=False)
        print(f"\tT-test: \tp_orig={pt_orig:.3e}, p_drawn={pt_ln:.3e}")

        _, pdiff_mwu_0 = mannwhitneyu(x0, x_ln0)
        _, pdiff_mwu_1 = mannwhitneyu(x1, x_ln1)
        _, pdiff_ttest_0 = ttest_ind(x0, x_ln0, equal_var=False)
        _, pdiff_ttest_1 = ttest_ind(x1, x_ln1, equal_var=False)
        print(f"\tdifference fit vs drawn? p_mwu={pdiff_mwu_0:.1g} / {pdiff_mwu_1:.1g}, p_ttest={pdiff_ttest_0:.1g} / {pdiff_ttest_1:.1g}")


    if plot:

        xvals = np.linspace(0.1, np.max(np.concatenate([x0, x1])), 1000).reshape(-1, 1)
        pdf_ln0 = lognorm.pdf(xvals, *p0)
        pdf_ln1 = lognorm.pdf(xvals, *p1)

        from scipy.stats import norm
        # pdf_n0 = norm.pdf(np.log(xvals), loc=np.mean(np.log(x0)), scale=p0[0])
        # pdf_n1 = norm.pdf(np.log(xvals), loc=np.mean(np.log(x1)), scale=p1[0])
        pdf_n0 = norm.pdf(np.log(xvals), loc=np.mean(np.log(p0[2] + p0[1])), scale=p0[0])
        pdf_n1 = norm.pdf(np.log(xvals), loc=np.mean(np.log(p1[2] + p1[1])), scale=p1[0])


        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].hist(x0, density=True, label="Original")
        ax[0].hist(x_ln0, density=True, alpha=0.5, label="drawn")
        ax[0].plot(xvals, pdf_ln0, ls=":", label="PDF", zorder=1)
        ax[1].hist(x1, density=True, label="original")
        ax[1].hist(x_ln1, density=True, alpha=0.5, label="drawn")
        ax[1].plot(xvals, pdf_ln1, ls=":", label="PDF", zorder=1)
        ax[0].legend()
        ax[0].set_title("Non-responder")
        ax[1].legend()
        ax[1].set_title("Responder")
        ax[1].set_xlim(0, 400)

        # plot underlying gaussians
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # ax[0].hist(np.log(x0), density=True, label="Original")
        # ax[0].hist(np.log(x_ln0), density=True, label="drawn", alpha=0.5, bins=20)
        # ax[0].plot(np.log(xvals), pdf_n0, ls=":", label="PDF", zorder=1)
        # ax[1].hist(np.log(x1), density=True, label="Original")
        # ax[1].hist(np.log(x_ln1), density=True, label="drawn", alpha=0.5, bins=20)
        # ax[1].plot(np.log(xvals), pdf_n1, ls=":", label="PDF", zorder=1)
        #
        # ax[0].legend()
        # ax[0].set_title("Non-responder")
        # ax[1].legend()
        # ax[1].set_title("Responder")

        plt.show()

    return x_ln0, x_ln1


def fit_lr_with_ci(x, y, Nboot=1000):
    from sklearn.utils import resample
    print(np.unique(y, return_counts=True))

    xvals = np.linspace(0, np.max(x), 100).reshape(-1, 1)
    lr0 = LogisticRegression(class_weight="balanced")
    lr0.fit(x, y)

    p0 = lr0.predict_proba(xvals)[:, -1]
    plt.plot(xvals, p0, "-", c="C1", lw=5)

    for r in range(Nboot):
        print(r)
        xr, yr = resample(x, y, replace=True, stratify=None)
        # xr, yr = resample(x, y, replace=False, n_samples=140)
        # print(xr.shape, yr.shape)
        # print(np.unique(yr, return_counts=True))

        lr = LogisticRegression(class_weight="balanced")
        lr.fit(xr, yr)

        p = lr.predict_proba(xvals)[:, -1]
        plt.plot(xvals, p, ":", color="black", alpha=0.1, zorder=-1)

    plt.yticks([0, .25, .50, .75, 1])
    plt.grid(1)
    plt.xlim(0, 300)
    # plt.title("LogNorm fit -> repeated subset sampling (N=140)")
    plt.title(f"Bootstrap original data (N={len(x)})")
    plt.show()

    pass


def plot_doseresponse_with_lognorm(x0, x1, xvals, phat_plot, title=""):
    fig, ax = plt.subplots()
    ax.plot(x0, [0] * len(x0), "o", label=f"Non-responder N={len(x0)}", ms=2)
    ax.plot(x1, [1] * len(x1), "o", label=f"Responder N={len(x1)}", ms=2)
    ax.plot(xvals, phat_plot, ":", c="black")
    ax.set_yticks([0, .25, .50, .75, 1])
    ax.grid(1)
    ax.set_xlim(0, 300)
    ax.legend()
    ax.set_title(title)
    plt.show()


    pass


def load_dose_volume_points(img_path="", plot=False):
    import cv2

    if not img_path:
        img_path = "../../PRRT_sample_size/figures/pub1_hebert24/dose_dtv_hebert_2024.png"

    # Pixel -> dose, volume-change convertion functions
    px_to_xval = np.array([[0, 70], [50, 205], [100, 340], [150, 474], [200, 609]])         # total tumor absorbed dose, x-pixel value
    px_to_yval = np.array([[0, 221], [10, 189], [20, 156], [40, 92], [60, 27]])                 # delta tumor volume, y-pixel value

    # Needs this for some reason
    px_to_xval = np.array([[y, x] for (x, y) in px_to_xval])
    px_to_yval = np.array([[y, x] for (x, y) in px_to_yval])


    lr_x = LinearRegression(fit_intercept=True)
    lr_x.fit(*[x.reshape(-1, 1) for x in px_to_xval.T])
    # print(lr_x.coef_, lr_x.score(*[x.reshape(-1, 1) for x in px_to_xval.T]))

    lr_y = LinearRegression(fit_intercept=True)
    lr_y.fit(*[y.reshape(-1, 1) for y in px_to_yval.T])
    # print(lr_y.coef_, lr_y.score(*[y.reshape(-1, 1) for y in px_to_yval.T]))

    # Load and process image from publication
    img = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = image_rgb[:, :, 0] # single channel (w best contrast)
    mask = cv2.inRange(gray_image, 0, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the x, y coordinates of each detected point
    points = []
    for contour in contours:
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    points = np.array(points)
    print("LOADED", points.shape, "points")

    points_x = lr_x.predict(points.T[0].reshape(-1, 1))
    points_y = lr_y.predict(points.T[1].reshape(-1, 1))

    print(f"\tdose-vals: \trange = {np.min(points_x):.1f} - {np.max(points_x):.1f} (mean = {np.mean(points_x):.1f})")
    print(f"\tvolume-change: range = {np.min(points_y):.1f} - {np.max(points_y):.1f} (mean = {np.mean(points_y):.1f})")

    if plot:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(points_x, points_y, "o", ms=2)
        ax[0].hlines([20], 0, 300, ls=":")
        ax[0].set_xlim(0, 300)
        ax[0].set_title(f"{len(points_x)} points found")
        ax[1].hist(points_x, bins=21)
        ax[1].set_xlabel("tTAD (Gy)")
        ax[0].set_ylabel("dTV")
        plt.show()

    return points_x, points_y


if __name__ == "__main__":



    pass
