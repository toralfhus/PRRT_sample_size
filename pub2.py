from utils_load import *
from utils_mc import *

nm = "pub2_warfvinge24"
folder_data = os.path.join(os.getcwd(), "data", nm)
folder_figs = os.path.join(os.getcwd(), "figures", nm)

df = load_pub2(folder_data)

print(df)

