from pathlib import Path
import pandas as pd
import numpy as np

DATASET_ROOT = Path(r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset")
ORI_POS = [22, 23, 24]  # 1-based: Yaw, Pitch, Roll

def read_csv(p):
    try:
        return pd.read_csv(p, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="cp1252")

def circ_diff(a,b):
    return (b-a+180.0)%360.0-180.0

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)
    sfiles=sorted([p for p in DATASET_ROOT.rglob("S-*.csv")])
    if not sfiles: raise RuntimeError("No S files found.")
    print("="*72)
    print("MAVeriCK ML2 - STEP 8: ORIGINAL PHONE ORIENTATION ANALYSIS")
    print("="*72)
    print(f"Smartphone files: {len(sfiles)}")
    print("Sampling rate: 10 Hz")
    print()

    all_pitch=[]; all_diffs=[]; pair_rows=[]; examples=[]
    total_samples=0; yaw_wrap=0; roll_wrap=0

    for p in sfiles:
        key=p.stem[2:].lower()
        df=read_csv(p)
        cols=[df.columns[i-1] for i in ORI_POS]
        o=df[cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        o=o[np.isfinite(o).all(axis=1)]
        if len(o)<2: continue
        yaw,pitch,roll=o.T
        dy_raw=np.abs(np.diff(yaw)); dy_c=np.abs(circ_diff(yaw[:-1],yaw[1:]))
        dp=np.abs(np.diff(pitch)); dr_raw=np.abs(np.diff(roll)); dr_c=np.abs(circ_diff(roll[:-1],roll[1:]))
        all_pitch.extend(pitch.tolist())
        all_diffs.extend(np.column_stack([dy_raw,dy_c,dp,dr_raw,dr_c]).tolist())
        yw=int(np.sum(dy_raw>180)); rw=int(np.sum(dr_raw>180))
        yaw_wrap+=yw; roll_wrap+=rw; total_samples+=len(o)
        pair_rows.append([key,len(o),pitch.min(),pitch.max(),np.mean(np.abs(pitch)),
                          np.sum(np.abs(np.abs(pitch)-90)<=5),
                          np.sum(np.abs(np.abs(pitch)-90)<=10),yw,rw])
        idx=np.where((dy_raw>180)|(dr_raw>180))[0]
        for i in idx[:3]:
            examples.append([key,i,yaw[i],yaw[i+1],pitch[i],pitch[i+1],roll[i],roll[i+1],
                              dy_raw[i],dy_c[i],dr_raw[i],dr_c[i]])

    pitch=np.asarray(all_pitch,float); d=np.asarray(all_diffs,float)
    names=["yaw_raw","yaw_circular","pitch_raw","roll_raw","roll_circular"]

    print("1. PITCH DISTRIBUTION")
    print("-"*72)
    for q in [0,1,5,25,50,75,90,95,99,100]:
        print(f"{q:5.1f} percentile : {np.percentile(pitch,q):.6f} deg")
    for t in [80,85,88,89]:
        n=np.sum(np.abs(pitch)>=t)
        print(f"|pitch| >= {t:2d} deg : {n:8d} ({100*n/len(pitch):.4f}%)")

    print("\n2. CONSECUTIVE 10 Hz ORIENTATION CHANGES")
    print("-"*72)
    for j,nm in enumerate(names):
        v=d[:,j]
        print(f"\n{nm}")
        for q in [50,90,95,99,99.9,100]:
            print(f"  {q:5.1f} percentile : {np.percentile(v,q):.6f} deg")

    print("\n3. EULER WRAP-AROUND INDICATORS")
    print("-"*72)
    print(f"Yaw consecutive raw jumps >180° : {yaw_wrap}")
    print(f"Roll consecutive raw jumps >180°: {roll_wrap}")
    if examples:
        print("\nFirst 12 wrap candidates (raw vs circular):")
        print("pair sample yaw0 yaw1 pitch0 pitch1 roll0 roll1 yaw_raw yaw_circ roll_raw roll_circ")
        for r in examples[:12]:
            print(f"{r[0]:8s} {r[1]:6d} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7.2f} "
                  f"{r[6]:8.2f} {r[7]:8.2f} {r[8]:8.2f} {r[9]:8.2f} {r[10]:8.2f} {r[11]:8.2f}")

    print("\n4. PER-PAIR SUMMARY")
    print("-"*72)
    summary=pd.DataFrame(pair_rows,columns=[
        "pair","samples","pitch_min_deg","pitch_max_deg","mean_abs_pitch_deg",
        "samples_within_5deg_of_90","samples_within_10deg_of_90",
        "yaw_raw_jumps_gt_180","roll_raw_jumps_gt_180"])
    print(summary.to_string(index=False))

    base=Path(__file__).resolve().parent
    summary.to_csv(base/"step8_orientation_pair_summary.csv",index=False)
    print("\n"+"="*72)
    print("STEP 8 COMPLETE")
    print("="*72)
    print(f"Samples analyzed: {total_samples}")
    print(f"Pair summary saved: {base/'step8_orientation_pair_summary.csv'}")
    print("No source, window, or target files were modified.")

if __name__=="__main__":
    main()
