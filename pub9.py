from utils_load import *
from utils_mc import *

pub_name = "pub9_jahn20"
folder_data = os.path.join(os.getcwd(), "data", pub_name)
folder_figs = os.path.join(os.getcwd(), "figures", pub_name)

do_calc = False
n_rep = 1000
n_min = 3
desired_power = 0.9

stat_func = pearsonr
stat_names = ["r", "p"]
stat_sign = lambda p1, p2: p1 ** 2 >= .25 and p2 < .05

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
ax = ax.ravel()


df = load_pub9(folder_data)
cx = "Cumulative absorbed dose (Gy)"

x = df[cx]

for i, cy in enumerate(["Tumour shrinkage diameter (%)", "Tumour shrinkage volume (%)"]):
    print("\n", cy)
    y = df[cy]

    nm = cy.split(" ")[2]
    path_save = os.path.join(folder_data, f"boot_{nm}.csv")

    if do_calc:
        compute_bootstrapped_params(x, y, stat_names, stat_func, path_save=path_save, n_min=n_min, n_rep=n_rep, nm=cy)

    # analyze_bootstrapped_params(path_save, stat_sign, nm=cy, plot=(fig, ax[i]), fit=True, desired_power=desired_power)
    analyze_bootsrapped_negative_results(path_save, nm=cy, plot=(fig, ax[i]), desired_power=desired_power)


# savefig_path = os.path.join(folder_figs, f"{pub_name}_power_boot.pdf")
savefig_path = os.path.join(folder_figs, f"{pub_name}_power_boot_equiv.pdf")
plt.savefig(savefig_path)
print("SAVED monte-carlo estimated type II error rates:", savefig_path)
