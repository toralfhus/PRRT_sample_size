from utils_load import *
from utils_mc import *
from scipy.stats import linregress

pub_name = "pub5_ha24"
folder_data = os.path.join(os.getcwd(), "data", pub_name)
folder_figs = os.path.join(os.getcwd(), "figures", pub_name)
os.makedirs(folder_figs, exist_ok=True)
os.makedirs(folder_data, exist_ok=True)


replicate_original = False
do_calc = False
n_rep = 1000
n_min = 3
desired_power = 0.9

stat_func = pearsonr
stat_names = ["r", "p"]
stat_sign = lambda r, p: p < .05 # only p <.05, no R2 threshold


if replicate_original:
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
    ax = ax.ravel()

    dfs = load_pub5(folder_data)

    for i, dfi in enumerate(dfs):
        x, y, nm = dfi
        print("\n", nm, x.shape, y.shape)


        sp_rho, sp_rho_p = pearsonr(x, y, alternative='two-sided')
        print(f"\tPEARSON: R2={sp_rho ** 2:.2f}, rho_p={sp_rho_p:.3f}")
        ax[i].plot(x, y, "o", label=None)
        ax[i].axhline(y=0, color="black", linestyle="-", alpha=0.5)

        a, b, lr_r2, lr_p, _ = linregress(x, y)
        lr_r2 *= lr_r2
        print(f"\tLINREG: R2={lr_r2:.2f}, lr_p={lr_p:.3f}")
        lrx = np.linspace(min(x), max(x), 100)
        lry = a * lrx + b
        ax[i].plot(lrx, lry, ":", label=None)

        ax[i].set_xlabel(x.name)
        ax[i].set_ylabel(y.name)
        ax[i].set_title(f"{nm}\nN={len(x)}, r2={sp_rho ** 2:.2f}, p={sp_rho_p:.3f}")
        # ax[i].legend(loc="best")

    fig.tight_layout(h_pad=5)
    savefig_path = os.path.join(folder_figs, f"pub5_ha24_replicated.pdf")
    plt.savefig(savefig_path)
    print("SAVED replications", savefig_path)


dfs = load_pub5(folder_data)
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
ax = ax.ravel()

# Iterate over the figures
for i, dfi in enumerate(dfs):

    x_orig, y_orig, nm = dfi
    path_save = os.path.join(folder_data, f"boot_{nm}.csv")

    if do_calc:
        compute_bootstrapped_params(x_orig, y_orig, stat_names, stat_func, path_save=path_save, n_min=3, n_rep=1000, nm=nm)
    analyze_bootstrapped_params(path_save, stat_sign, nm=nm, plot=(fig, ax[i]), fit=True, desired_power=0.9)

fig.tight_layout(h_pad=5)
savefig_path = os.path.join(folder_figs, "power_boot.pdf")
plt.savefig(savefig_path)
print("SAVED computed type II error rates:", savefig_path)
