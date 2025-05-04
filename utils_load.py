from utils import *


def load_pub1(folder, categorical=False, plot=False):
    from dev_utils_mc import load_dose_volume_points

    points_x, points_y = load_dose_volume_points(img_path=os.path.join(folder, "dose_dtv_hebert_2024.png"), plot=plot)

    if categorical:
        x0 = points_x[points_y > 0]
        x1 = points_x[points_y <= 0]
        print("Class balance non-responder / responder:", len(x0), len(x1))

        x = np.concatenate([x0, x1]).reshape(-1, 1)  # tTAD
        y = np.array([0] * len(x0) + [1] * len(x1)).reshape(-1, 1)
        return x, y
    else:
        return np.ravel(points_x), np.ravel(points_y)


def load_pub2(folder):
    print(f"\n--- Loading data from Figures 2/3 A/B/C from pub5: Ha et al, EJNMMI Physics, 2024 ---")
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub2" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)

    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)
        nm = f.split(".")[0]    # remove .csv
        print(df)
        print("not finished")
        sys.exit()

        return df.T.iloc[0], df.T.iloc[1], nm

    pass


def load_pub3(folder):
    print(f"\n--- Loading data from Figure 3 from pub3: Maccauro al, ?, ? ---")
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub3" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)

    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)
        nm = f.split(".")[0]    # remove .csv
        print(df)
        print("not finished")
        sys.exit()

        return df.T.iloc[0], df.T.iloc[1], nm

    pass


def load_pub4(folder):
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub4" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)
    print()
    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)

        nm = f.split(".")[0]    # remove .csv
        yield df.T.iloc[0], df.T.iloc[1], nm


def load_pub5(folder):
    # Ha et al, EJNMMI Physics, 2024 https://doi.org/10.1186/s40658-024-00620-8
    print(f"\n--- Loading data from Figures 2/3 A/B/C from pub5: Ha et al, EJNMMI Physics, 2024 ---")
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub5" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)
    print()
    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)

        nm = f.split(".")[0]    # remove .csv
        yield df.T.iloc[0], df.T.iloc[1], nm

    pass


def load_pub6(folder):
    files = os.listdir(folder)
    files = list(filter(lambda f: ".csv" in f and "Pub6" in f, files))
    files = list(filter(lambda f: "boot" not in f, files))

    files.sort()
    print(files)
    print()
    for f in files:
        print("\tLOADING", f, end=" ")
        df = pd.read_csv(os.path.join(folder, f), index_col=None, encoding='unicode_escape')   # no pt indexing in csv
        print(df.shape, df.columns.values)

        nm = f.split(".")[0]    # remove .csv
        yield df.T.iloc[0], df.T.iloc[1], nm


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

