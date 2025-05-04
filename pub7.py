from utils_load import *
from utils_mc import *

folder_data = os.path.join(os.getcwd(), "data", "pub7_jahn21")
folder_figs = os.path.join(os.getcwd(), "figures", "pub7_jahn21")

dfs = load_pub7(folder_data)

# Replicate figures 6A / 6B / 7A / 7B using extracted data
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 12))
ax = ax.ravel()

for i, dfi in enumerate(dfs):
    x, y, nm = dfi
    # print(x.shape, y.shape)
    rho, rho_p = pearsonr(x, y, alternative='two-sided')
    print(f"\tR2={rho ** 2:.2f}, rho_p={rho_p:.3f}")
    ax[i].plot(x, y, "o", label=f"N={len(x)}, r2={rho ** 2:.2f}, p={rho_p:.3f}")
    ax[i].set_xlabel(x.name)
    ax[i].set_ylabel(y.name)
    ax[i].set_title(nm)
    ax[i].legend(loc="best")

savefig_path = os.path.join(folder_figs, "pub7_jahn21_replicated.png")
plt.savefig(savefig_path)
print("SAVED replications of figures 6/7 A/B", savefig_path)


# Monte Carlo type II error rate calculation using bootstraps
dfs = load_pub7(folder_data)

do_calc = False     # load if false


stat_func = pearsonr
stat_names = ["r", "p"]  # statistical parameters to evaluate using param_func
# stat_sign = [lambda p: p**2>0.25, lambda p: p<0.05]
stat_sign = lambda p1, p2: p1 ** 2 >= .25 and p2 < .05

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 14))
ax = ax.ravel()

# Iterate over the four figures
for i, dfi in enumerate(dfs):

    x_orig, y_orig, nm = dfi
    path_save = os.path.join(folder_data, f"boot_{nm}.csv")

    if do_calc:
        compute_bootstrapped_params(x_orig, y_orig, stat_names, stat_func, path_save=path_save, n_min=3, n_rep=1000, nm=nm)
    analyze_bootstrapped_params(path_save, stat_sign, nm=nm, plot=(fig, ax[i]), fit=True, desired_power=0.9)

savefig_path = os.path.join(folder_figs, f"{nm}_power_boot.pdf")
plt.savefig(savefig_path)
print("SAVED computed type II error rates:", savefig_path)

