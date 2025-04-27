import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score, precision_score, mean_squared_error
from scipy.stats import mannwhitneyu, ttest_ind, lognorm

from dev_utils_mc import *
from dev_extract_points import load_dose_volume_points

points_x, points_y = load_dose_volume_points()
x0 = points_x[points_y > 0]
x1 = points_x[points_y <= 0]
x = np.concatenate([x0, x1]).reshape(-1, 1)
xvals = np.linspace(0, np.max(x), 100).reshape(-1, 1)
y = np.array([0] * len(x0) + [1] * len(x1)).reshape(-1, 1)

# FIT lognorm distributions -> use instead of raw data?
# x_ln0, x_ln1 = fit_and_draw_lognorms(x0, x1, Ndraw=10000, plot=True)
# sys.exit()
# x0 = x_ln0; x1 = x_ln1
# x = np.concatenate([x_ln0, x_ln1]).reshape(-1, 1)
# y = np.array([0] * len(x_ln0) + [1] * len(x_ln1)).reshape(-1, 1)
# sys.exit()

# fit_lr_with_ci(x, y, Nboot=1000)

# lr0 = LogisticRegression()
lr0 = LogisticRegression(class_weight="balanced")
lr0.fit(x, y)
phat = lr0.predict_proba(x)[:, -1]
phat_plot = lr0.predict_proba(xvals)[:, -1]

# plot_doseresponse_with_lognorm(x0, x1, xvals, phat_plot)


p_thresh_log = 0.5  # TODO: ROC-analysis?
alpha = 0.05    # type I error rate
beta = 0.1      # type II / 1 - power

# Monte carlo or simple random subset sampling (SRS) (+ bootstrapping?)
# savename = f"mc_lognorm_p0mu0=[{s0:.2f}, {mu0:.2f}]_p1mu1=[{s1:.2f}, {mu1:.2f}]_shuff.csv"

# TODO: real points -> fit log-normal -> monte-carlo
# TODO: plot isocontour-lines for population estimate (z) given xy = log-alpha / log-beta ?
p0, p1 = fit_and_draw_lognorms(x0, x1, return_params=True)
# savename = f"mc_hebert2024.csv"
# monte_carlo_sample_size_calculate(sample_func=lognorm.rvs, params_0=p0, params_1=p1, Nmax=200, num_repeat=1000, shuffle=False, savename=savename)
# savename = f"mc_hebert2024_shuff.csv"
# monte_carlo_sample_size_calculate(sample_func=lognorm.rvs, params_0=p0, params_1=p1, Nmax=200, num_repeat=1000, shuffle=True, savename=savename)
# monte_carlo_sample_size_calculate(sample_func=np.random.lognormal, params_0=[s0, mu0], params_1=[s1, mu1], Nmax=200, num_repeat=1000, shuffle=True, savename=savename)
# monte_carlo_sample_size_analyze_typeII(savename, alpha=0.05, power=0.9)


# savename = f"rsb_hebert2024_strat=False.csv"
# repeated_bootstrap_samplesize_calculate(x, y, num_repeat=1000, stratify=False, shuffle=False, savename=savename)

savename = f"rsb_hebert2024_strat=True.csv"
repeated_bootstrap_samplesize_calculate(x, y, num_repeat=1000, stratify=True, shuffle=False, savename=savename)

sys.exit()


roc_analysis(y, phat)

PPV, TPR, TNR, NPV, AUC, BS, ACC = get_logistic_prediction_scores(y, phat, p_thresh=p_thresh_log)
mwu = mannwhitneyu(x0, x1)
U, p_mwu = float(mwu.statistic), float(mwu.pvalue)
print(f"\tMWU: U={U:.3e}, p={p_mwu:.3e}")
T, p_t = ttest_ind(x0, x1, equal_var=False)
print(f"\tT-test: T={T:.3e}, p={p_t:.3e}")

# srs_samplesize_calculate(x, y) #TODO




# Convert pixel measurements from prev publication to Gy
x_scale_gy = [0, 100, 200, 300]
x_scale_px = [148, 400, 653, 906]
lin = LinearRegression(fit_intercept=True)
lin.fit(np.array(x_scale_px).reshape(-1, 1), np.array(x_scale_gy).reshape(-1, 1))
a, b = float(lin.coef_), float(lin.intercept_)
print(f"Px to gy = {a:.2f}x + {b:.2f}")
px_to_gy = lambda x: a*x + b
# plt.plot(x_scale_px, x_scale_gy, "x")
# plt.plot(x_scale_px, px_to_gy(np.array(x_scale_px)), ":")
# plt.show()
x_list = np.array([201, 725, 874, 546, 464])    # response
print("Gy resp:", px_to_gy(x_list))
x_list = np.array([169, 492, 806, 494, 330])    # no response
print("Gy noresp:", px_to_gy(x_list))

global lr_gy_vals, lr_pc_vals
lr_gy_vals = px_to_gy(np.array([199, 351, 470, 906])).reshape(-1, 1)
lr_pc_vals = np.array([0.5, 0.75, 0.875, 0.99]).reshape(-1, 1)
print(lr_gy_vals.ravel())
lr0 = LogitRegression()

lr0.fit(lr_gy_vals, lr_pc_vals)
print(lr0.intercept_, lr0.coef_)
print(lr0.predict(lr_gy_vals).ravel())

N = 1000
N_train = 1000
N_test = 1000

# mu0, s0 = 4, .4
# mu1, s1 = 4.5, .3
# mu0, mu1, s0, s1 = 4.32, 4.84, 0.93, 0.75   # with outliers
# mu0, mu1, s0, s1 = 4.32, 4.84, 0.93, 0.75   # with outliers

mu0, s0 = estimate_lognorm_params(72, 8, 136)
mu1, s1 = estimate_lognorm_params(126, 21, 228)
dosefunc = np.random.lognormal



# mu0, mu1 = 72, 126
# s = 0.5
# s0, s1 = s, s
# mu0 *= 1 / s0
# mu1 *= 1 / s1
# dosefunc = np.random.binomial


print(mu0, s0, mu1, s1)



dmax = 300
dmin = -dmax    # high enough range to not bias comparisons of probability-distributions (MSE, BS etc)

xvals = np.linspace(dmin, dmax, N).reshape(-1, 1)
pvals = lr0.predict(xvals).reshape(-1, 1)
print(pvals.shape)

# print(mean_squared_error(np.array([0]*len(pvals)), pvals))    # check bias
# print(mean_squared_error(np.array([0.5]*len(pvals)), pvals))
# print(mean_squared_error(np.array([1]*len(pvals)), pvals))


# search_lognorm_params_to_logistic_fit_old(x_true=xvals, p_true=pvals, z0=2.33, N_fit=N//2, max_iter=10)

# _ = search_lognorm_params_to_logistic_fit(xvals, pvals, init_params=[mu0, s0, mu1, s1])
# mu0, s0, mu1, s1 = search_lognorm_params_to_logistic_fit(xvals, pvals, init_params=[mu0, s0, mu1, s1])
# mu0, s0, mu1, s1 = search_lognorm_params_to_logistic_fit(xvals, pvals, init_params=[0.1, 0.1, 0.1, 0.1])
# mu0, s0, mu1, s1 = search_lognorm_params_to_logistic_fit(xvals, pvals, init_params=[3, 0.5, 4, 0.5])    # SE=17.106 (SE0=40.255)
# mu0, s0, mu1, s1 = search_lognorm_params_to_logistic_fit(xvals, pvals, init_params=[4, 0.5, 4.5, 0.5])    #
# mu0, s0, mu1, s1 = search_lognorm_params_to_logistic_fit(pvals, init_params=[2, 1, 2, 1])

# mu0, s0, mu1, s1 = [4.27666566, 0.60798628, 4.83628269, 0.51176501]     # MSE=0.020
# mu0, s0, mu1, s1 = [4.0, 0.5, 4.8, 0.5]     # MSE=0.020

# mu0, s0, mu1, s1 = [-8.45937949, 0.8492827, -1.518034, 0.78412331]    # SE= 5032 MSE=0.053

# sys.exit()


#####


p_thresh_log = 0.5  # TODO: find optimal logistic threshold using ROC? or to achieve 80% accuracy?


# Analytic sample size estimation
# TODO: try T-test on log transformed -> same result?
from scipy.stats import lognorm, norm

alpha = 0.05    # type I error rate
beta = 0.1      # type II / 1 - power

mean_0, mean_1 = lognorm.mean(s=s0, loc=0, scale=np.exp(mu0)), lognorm.mean(s=s1, loc=0, scale=np.exp(mu1))
sd0, sd1 = lognorm.std(s=s0, loc=0, scale=np.exp(mu0)), lognorm.std(s=s1, loc=0, scale=np.exp(mu1))
analytic_sample_size_estimate(mean_0, mean_1, sd0, sd1, alpha=alpha, beta=beta)
analytic_sample_size_estimate(mean_0, mean_1, sd0, sd1, alpha=0.01, beta=0.01)
analytic_sample_size_estimate(mean_0, mean_1, sd0, sd1, alpha=0.001, beta=0.001)
# TODO: plot isocontour-lines for population estimate (z) given xy = log-alpha / log-beta ?

# fig, ax = plt.subplots()
# x = np.linspace(lognorm.ppf(0.001, s0, scale=np.exp(mu0)),
#                 lognorm.ppf(0.999, s0, scale=np.exp(mu0)), 100)
# ax.plot(x, lognorm.pdf(x, s0, scale=np.exp(mu0)),
#        'r-', lw=5, alpha=0.6, label='lognorm pdf')
# x1 = np.linspace(lognorm.ppf(0.001, s1, scale=np.exp(mu1)),
#                 lognorm.ppf(0.999, s1, scale=np.exp(mu1)), 100)
# ax.plot(x1, lognorm.pdf(x1, s1, scale=np.exp(mu1)),
#        'b-', lw=5, alpha=0.6, label='lognorm pdf')
# plt.show()

# savename = f"mc_lognorm_p0mu0=[{s0:.2f}, {mu0:.2f}]_p1mu1=[{s1:.2f}, {mu1:.2f}].csv"
savename = f"mc_lognorm_p0mu0=[{s0:.2f}, {mu0:.2f}]_p1mu1=[{s1:.2f}, {mu1:.2f}]_shuff.csv"
# monte_carlo_sample_size_calculate(sample_func=np.random.lognormal, params_0=[s0, mu0], params_1=[s1, mu1], Nmax=200, num_repeat=1000, shuffle=True, savename=savename)
# monte_carlo_sample_size_calculate(sample_func=np.random.lognormal, params_0=[s0, mu0], params_1=[s1, mu1], Nmax=200, num_repeat=1000, shuffle=True, savename=savename)


monte_carlo_sample_size_analyze_typeII(savename, alpha=0.05, power=0.9)
# monte_carlo_sample_size_analyze_typeII(savename, alpha=0.01, power=0.99)
# monte_carlo_sample_size_analyze_typeII(savename, alpha=0.001, power=0.999)

# monte_carlo_sample_size_analyze_typeI(savename, power=0.9, alpha=0.05)

sys.exit()


# mu0, mu1, s0, s1 = 5, 6, 1, 1
# mu0, mu1, s0, s1 = 75, 127, 21, 34   # mu = center of mass, sd = range / 6 -> wo outlier
# mu0, mu1, s0, s1 = 139, 159, 41, 44   # with outliers
# dosefunc = np.random.normal

noresp_train = dosefunc(mu0, s0, size=N_train)
resp_train = dosefunc(mu1, s1, size=N_train)

noresp_test = dosefunc(mu0, s0, size=N_test)
resp_test = dosefunc(mu1, s1, size=N_test)


# df1_train = pd.DataFrame(data=np.array([resp_train, np.repeat(1, len(resp_train))]).T, columns=["ad", "resp"], index=range(1, 1 + len(resp_train)))
df0_train = pd.DataFrame(data=np.array([noresp_train, np.repeat(0, N_train), np.repeat(1, N_train)]).T, columns=["ad", "resp", "train"], index=range(len(resp_train) + 1, 1 + len(resp_train) + len(noresp_train)))
df1_train = pd.DataFrame(data=np.array([resp_train, np.repeat(1, N_train), np.repeat(1, N_train)]).T, columns=["ad", "resp", "train"], index=range(1, 1 + len(resp_train)))

df_train = pd.concat([df0_train, df1_train], axis=0).sort_values(by=["resp", "ad"], ascending=True)
# print(df_train)

df0_test = pd.DataFrame(data=np.array([noresp_test, np.repeat(0, N_test), np.repeat(0, N_test)]).T, columns=["ad", "resp", "train"], index=range(len(resp_test) + 1, 1 + len(resp_test) + len(noresp_test)))
df1_test = pd.DataFrame(data=np.array([resp_test, np.repeat(1, N_test), np.repeat(0, N_test)]).T, columns=["ad", "resp", "train"], index=range(1, 1 + len(resp_test)))
df_test = pd.concat([df0_test, df1_test], axis=0).sort_values(by=["resp", "ad"], ascending=True)

df = pd.concat([df_train, df_test])
print(df)


x_train, y_train = df_train["ad"].values.reshape(-1, 1), df_train["resp"].values.reshape(-1, 1)
x_test, y_test = df_test["ad"].values.reshape(-1, 1), df_test["resp"].values.reshape(-1, 1)
print(x_train.shape, y_train.shape)
# sys.exit()
median_train = np.median(df_train["ad"])
dmin, dmax, dsd = np.min(df["ad"]), np.max(df["ad"]), np.std(df["ad"])

# print(f"median train = {median_train:.2f} Gy, range=({dmin:.2f}, {dmax:.2f}), sd={dsd:.2f}")
print(f"train nonresp ->\trange=({np.min(df0_train['''ad''']):.2f}, {np.max(df0_train['''ad''']):.2f}), median={np.median(df0_train['''ad''']):.2f}, mean={np.mean(df0_train['''ad''']):.2f}, std={np.std(df0_train['''ad''']):.2f}")
print(f"train resp ->\t\trange=({np.min(df1_train['''ad''']):.2f}, {np.max(df1_train['''ad''']):.2f}), median={np.median(df1_train['''ad''']):.2f}, mean={np.mean(df1_train['''ad''']):.2f}, std={np.std(df1_train['''ad''']):.2f}")


lr = LogisticRegression(penalty=None, fit_intercept=True)
lr.fit(x_train, y_train)

phat_train = lr.predict_proba(x_train)[:, -1]
yhat_train = lr.predict(x_train)
print(yhat_train.shape)

phat_test = lr.predict_proba(x_test)[:, -1]


phat_fit = lr.predict_proba(xvals)[:, -1]
mse = mean_squared_error(pvals, phat_fit)
print(f"MSE={mse:.3f}")

# print(lr.classes_)
# print(yhat.shape)
bs, auc, rec, prec = (brier_score_loss(y_train, phat_train), roc_auc_score(y_train, phat_train),
                      recall_score(y_train, yhat_train), precision_score(y_train, yhat_train))


print("\n-- TRAIN --")
PPV, TPR, TNR, NPV, AUC, BS, ACC = get_logistic_prediction_scores(y_train, phat_train, p_thresh=p_thresh_log)
# mwu = mannwhitneyu(x_train, y_train)
mwu = mannwhitneyu(df0_train["ad"], df1_train["ad"])
U, p_mwu = float(mwu.statistic), float(mwu.pvalue)
print(f"\tMWU: U={U:.3e}, p={p_mwu:.3e}")
T_train, p_train = ttest_ind(df0_train["ad"], df1_train["ad"], equal_var=False)
print(f"\tT-test: T={T_train:.3e}, p={p_train:.3e}")

print("\n-- TEST --")
get_logistic_prediction_scores(y_test, phat_test, p_thresh=p_thresh_log)
mwu_test = mannwhitneyu(df0_test["ad"], df1_test["ad"])
U_test, p_test = float(mwu_test.statistic), float(mwu_test.pvalue)
print(f"\tMWU: U={U_test:.3e}, p={p_test:.3e}")
T_test, p_ttest = ttest_ind(df0_test["ad"], df1_test["ad"], equal_var=False)
print(f"\tT-test: T={T_test:.3e}, p={p_ttest:.3e}")


# xvals = np.linspace(0, dmax, int(dmax)).reshape(-1, 1)
# xvals = np.linspace(0, N-1, N).reshape(-1, 1)

# yhat_test = lr.predict_proba(x_test)[:, -1]
# bs_test, auc_test = brier_score_loss(y_test, yhat_test), roc_auc_score(y_test, yhat_test)


fig, ax = plt.subplots(ncols=3, figsize=(16, 6)); ax=ax.ravel()
sns.histplot(data=df_train, x="ad", hue="resp", ax=ax[0])
ax[0].set_title("TRAIN")

sns.histplot(data=df, x="ad", hue="train", ax=ax[1])
ylims = ax[1].get_ylim()
ax[1].vlines([np.median(df_train["ad"])], *ylims, ls=":", color="black")
ax[1].set_ylim(ylims)
ax[1].set_title(f"mu0, s0, mu1, s1 = {np.round([mu0, s0, mu1, s1], 2)}")

# logistic
sns.scatterplot(data=df, x="ad", y="resp", hue="train", ax=ax[2], linewidth=0, alpha=0.5)
ax[2].plot(xvals, phat_fit, "-", color="black", alpha=0.5, ms=2, zorder=1, label="LR_fit")
ax[2].plot(xvals, lr0.predict(xvals).ravel(), ":", color="black", alpha=0.5, label="LR_0")
ax[2].set_xlim(0, 300)
ax[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax[2].legend()
ax[2].grid()
ax[2].set_title(f"MSE={mse:.3f}, bs={BS:.3f}, acc={ACC}, auc={AUC:.3f}")
plt.show()
