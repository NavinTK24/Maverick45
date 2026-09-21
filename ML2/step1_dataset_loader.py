from pathlib import Path
import sys
import pandas as pd

def find_pairs(root: Path):
    s_files, v_files = {}, {}
    for p in root.rglob("*.csv"):
        if p.name.startswith("S-"):
            s_files[p.stem[2:].lower()] = p
        elif p.name.startswith("V-"):
            v_files[p.stem[2:].lower()] = p
    keys = sorted(set(s_files) & set(v_files))
    return [(k, s_files[k], v_files[k]) for k in keys], s_files, v_files

def inspect_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # Some IO-VNBD CSV files contain Windows-1252/Latin-1 bytes.
        # Fall back only for reading; the original CSV is not modified.
        df = pd.read_csv(path, encoding="cp1252")
    print(f"\nFILE: {path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    for i, c in enumerate(df.columns, 1):
        print(f"{i:02d}. {c}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    return df

def main():
    if len(sys.argv) != 2:
        print('Usage: python step1_dataset_loader.py "PATH_TO_CATEGORISED_IOVNB_DATASET"')
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    print("=" * 70)
    print("MAVeriCK ML2 - STEP 1: SYNCHRONIZED S/V DATASET LOADER")
    print("=" * 70)
    print("Using only the selected Categorised IOVNB Dataset:")
    print(root)

    pairs, s_files, v_files = find_pairs(root)
    print(f"\nMatched S/V pairs: {len(pairs)}")
    print(f"S-only files: {len(set(s_files)-set(v_files))}")
    print(f"V-only files: {len(set(v_files)-set(s_files))}")

    print("\nMATCHED PAIRS")
    for key, s, v in pairs:
        print(f"{key:18s} | {s.name:20s} | {v.name}")

    if not pairs:
        print("\nNo matched pairs found.")
        return

    key, s_path, v_path = pairs[0]
    print("\n" + "=" * 70)
    print(f"INSPECTING FIRST PAIR: {key}")
    print("=" * 70)

    s_df = inspect_csv(s_path)
    v_df = inspect_csv(v_path)

    print("\n" + "=" * 70)
    print("BASIC CHECK")
    print("=" * 70)
    print(f"S rows: {len(s_df)}")
    print(f"V rows: {len(v_df)}")
    print("\nThe dataset documentation states that simultaneously collected S/V")
    print("datasets were manually synchronized. We will not invent a timestamp")
    print("alignment rule here.")
    print("\nSTEP 1 COMPLETE.")
    print("Next: verify the actual headers, then create 10-sample x 6-feature windows.")

if __name__ == "__main__":
    main()
