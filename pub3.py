from utils_load import *

pub_name = "pub3_maccauro"
folder_data = os.path.join(os.getcwd(), "data", pub_name)
folder_figs = os.path.join(os.getcwd(), "figures", pub_name)
os.makedirs(folder_figs, exist_ok=True)
os.makedirs(folder_data, exist_ok=True)

load_pub3(folder_data)

