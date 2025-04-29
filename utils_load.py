from utils import *

def load_pub7(folder):
    # Jahn et al, Cancers, 2021 https://doi.org/10.3390/cancers13050962
    # df with two columns: Accumulated absorbed dose until BR (Gy), Fractional shrinkage of diameter

    print(f"\n--- Loading data from Fig 6A/B + 7A/B from pub7: Jahn et al, Cancers, 2021 ---")
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub7" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)

    print()
    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)

        nm = f.split(".")[0]    # remove .csv
        # nm = "pub7_jahn21"
        yield df.T.iloc[0], df.T.iloc[1], nm

    pass


def load_pub9(folder):

    print(f"\n--- Loading tables 2 and 3 from pub9: Jahn et al 2020 ---")
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub9" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))
    print(files)


    if len(files) != 1:
        print("ERR: multiple files found, only one expected:", files)
        return 0

    else:
        f = files[0]
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')
        print("LOADED:", df.shape, df.columns.values)
        return df

