from scipy.stats import spearmanr, pearsonr

from utils_load import *
from utils_mc import *

pub_name = "pub1_hebert24"
folder_data = os.path.join(os.getcwd(), "data", pub_name)
folder_figs = os.path.join(os.getcwd(), "figures", pub_name)
os.makedirs(folder_figs, exist_ok=True)
os.makedirs(folder_data, exist_ok=True)

x_orig, y_orig = load_pub1(folder_data)

print(x_orig.shape, y_orig.shape)


replicate_original = False
do_calc = False
n_rep = 1000
n_min = 3
desired_power = 0.9

stat_func = pearsonr
stat_names = ["r", "p"]
stat_sign = lambda r, p: p < .05 # only p <.05, no R2 threshold


rho, p = spearmanr(x_orig, y_orig)
print("spearman:", f"rho={rho:.3f}, p={p:.3g}")

rho, p = pearsonr(x_orig, y_orig)
print("pearson:", f"rho={rho:.3f}, p={p:.3g}")

fig, ax = plt.subplots(figsize=(8, 8))
# ax = ax.ravel()
nm = "Pub1_Hebert_Fig3"

path_save = os.path.join(folder_data, f"boot_{nm}.csv")

if do_calc:
    compute_bootstrapped_params(x_orig, y_orig, stat_names, stat_func, path_save=path_save, n_min=3, n_rep=1000, nm=nm)
analyze_bootstrapped_params(path_save, stat_sign, nm=nm, plot=(fig, ax), fit=True, desired_power=0.9)

fig.tight_layout(h_pad=5)
savefig_path = os.path.join(folder_figs, "power_boot.pdf")
plt.savefig(savefig_path)
print("SAVED computed type II error rates:", savefig_path)
