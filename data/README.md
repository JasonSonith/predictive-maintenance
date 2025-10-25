# Datasets (DVC Guide)

We use **DVC (Data Version Control)** so big files (datasets) don’t live in GitHub.  
Git tracks tiny pointer files; DVC puts the real data in a storage location you choose.  
Result: small repo, easy sync across your computers.

---

## What you need

- **Git** and **Python 3.11+**
- Install DVC on each machine:

```powershell
# Windows (PowerShell)
python -m pip install --upgrade pip
pip install dvc
```

```bash
# macOS / Linux
python3 -m pip install --upgrade pip
pip3 install dvc
```

---
## Upload the data to that remote
```powershell
dvc push
```

Then push your normal Git changes:
```powershell
git push
```

---

## Get datasets on any other computer

1) Clone the repo and install DVC
```powershell
git clone <YOUR_REPO_URL> predictive-maintenance
cd predictive-maintenance
python -m pip install dvc
```

2) If you used a local/sync folder remote, point DVC to its path on this computer
```powershell
# Example if the storage folder sits beside the repo again
dvc remote modify storage url ..\_dvc_storage
```

3) Pull the data
```powershell
dvc pull
```

4) Check status
```powershell
dvc status   # should say "Data and pipelines are up to date."
```

That’s it—`data/raw/...` gets filled in.

---

## Daily use

- When you add or change files inside `data/raw/...`, run:
```powershell
dvc add data/raw/ai4i
git add data/raw/ai4i.dvc
git commit -m "Update AI4I data"
dvc push
git push
```

- On other machines: `git pull` then `dvc pull`.

---

## Folder layout (data)

```
data/
  raw/
    ai4i/
    cmapss/
    cwru/
      drive_end_12k/
      normal_baseline/
    ims/
      1st_test/
  clean/        # made by your scripts (ignored by Git)
  features/     # made by your scripts (ignored by Git)
```

> Big folders (`raw/`, `clean/`, `features/`) are **ignored by Git**.  
> Only `.dvc` files and DVC config are tracked.

---

## Source links (if you ever re-download manually)

- **IMS Bearings (Test 1)** — <https://data.nasa.gov/docs/legacy/IMS.zip>  
  Put extracted `1st_test` into `data/raw/ims/`
- **AI4I 2020 (UCI)** — <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>  
  Put into `data/raw/ai4i/`
- **CWRU Bearing Data Center** — <https://engineering.case.edu/bearingdatacenter>  
  Put into `data/raw/cwru/drive_end_12k/` and `.../normal_baseline/`
- **NASA C-MAPSS** — <https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/>  
  Put into `data/raw/cmapss/`

---

## Troubleshooting (plain English)

- **“bad DVC file name is git-ignored”**  
  Your `.gitignore` blocks the `.dvc` file. Make sure it has:
  ```
  data/raw/**         # ignore data
  !data/raw/**/       # allow directories
  !data/raw/**/*.dvc  # allow DVC pointer files
  ```
  Then run `dvc add …` and `git add …` again.

- **“dvc: command not found”**  
  Activate your virtual env or run as a module:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  # or
  python -m dvc --version
  ```

- **Switching remotes later**  
  Change where data lives without touching code:
  ```powershell
  dvc remote modify storage url s3://new-bucket/path
  git commit -am "Switch DVC remote"
  dvc push
  ```

---

## Quick copy-paste

```powershell
# New machine: pull datasets
git clone <YOUR_REPO_URL> predictive-maintenance
cd predictive-maintenance
python -m pip install dvc
dvc remote modify storage url ..\_dvc_storage   # if using local storage
dvc pull
```

```powershell
# After adding/updating files in data/raw/...
dvc add data/raw/ims/1st_test
git add data/raw/ims/1st_test.dvc
git commit -m "Update IMS test1 data"
dvc push
git push
```

**Reminder:** never commit big raw data to GitHub. Use DVC pointer files + `dvc push/pull`.
