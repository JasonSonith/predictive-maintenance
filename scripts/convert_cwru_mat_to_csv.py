import argparse, pathlib, re
import numpy as np
import pandas as pd

def load_mat(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def read(name):
                obj = f[name]
                a = np.array(obj)
                # h5py stores 2D column vectors as shape (1, N) or (N, 1); squeeze
                return np.array(a).squeeze()
            def walk(g, prefix=""):
                for k, v in g.items():
                    name = f"{prefix}{k}"
                    if isinstance(v, h5py.Dataset):
                        out[name] = read(name)
                    elif isinstance(v, h5py.Group):
                        walk(v, name + "/")
            walk(f, "")
        return out

def extract_series(d):
    out = {}
    keys = list(d.keys())

    # Case 1: direct channels in keys
    chan_like = [k for k in keys if any(s in k.lower() for s in ["de_time", "fe_time", "ba_time", "rpm"])]
    if chan_like:
        for k in chan_like:
            arr = np.asarray(d[k]).squeeze()
            out[k] = arr
        return out

    # Case 2: single key containing a struct/record
    if len(keys) == 1:
        k = keys[0]
        v = d[k]
        # SciPy loads MATLAB struct as np.void/structured array with dtype fields
        if hasattr(v, "dtype") and v.dtype.names:
            for field in v.dtype.names:
                arr = np.asarray(v[field]).squeeze()
                out[field] = arr
            return out

    # Fallback: take any 1D numeric arrays
    for k, v in d.items():
        a = np.asarray(v).squeeze()
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number) and a.size > 10:
            out[k] = a
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Folder with .mat files")
    p.add_argument("--out", dest="outp", required=True, help="Output folder for CSVs")
    args = p.parse_args()

    in_dir = pathlib.Path(args.inp)
    out_dir = pathlib.Path(args.outp)
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = sorted(in_dir.glob("*.mat"))
    if not mats:
        print(f"No .mat files found in {in_dir}")
        return

    for f in mats:
        print(f"Converting {f.name} ...")
        d = load_mat(str(f))
        channels = extract_series(d)

        if not channels:
            print(f"  Warning: no signals detected in {f.name}")
            continue

        for name, arr in channels.items():
            arr = np.asarray(arr).squeeze()

            # Skip non-series signals (scalars or length < 2)
            if arr.ndim == 0 or arr.size < 2:
                print(f"  (skip {name}: scalar/empty, size={arr.size})")
                continue

            # Flatten any 2D vectors to 1D
            if arr.ndim > 1:
                arr = arr.reshape(-1)

            # clean channel name for filename
            safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
            out_csv = out_dir / f"{f.stem}__{safe}.csv"
            df = pd.DataFrame({"value": arr})
            df.to_csv(out_csv, index_label="index")
            print(f"  -> {out_csv}")

if __name__ == "__main__":
    main()
