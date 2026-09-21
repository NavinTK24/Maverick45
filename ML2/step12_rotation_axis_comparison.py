import os, glob
import numpy as np
import pandas as pd

ROOT=r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
OUT=r"D:\Maverick\ML2\step12_rotation_axis_comparison.csv"
GYRO=[16,17,18]; ORI=[22,23,24]; DT=0.1; W=10

def norm(q):
    return q/max(np.linalg.norm(q),1e-12)
def qm(a,b):
    aw,ax,ay,az=a; bw,bx,by,bz=b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw])
def qi(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def e2q(y,p,r):
    y,p,r=np.deg2rad([y,p,r]); cy,sy=np.cos(y/2),np.sin(y/2); cp,sp=np.cos(p/2),np.sin(p/2); cr,sr=np.cos(r/2),np.sin(r/2)
    return norm(np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]))
def rvq(rv):
    a=np.linalg.norm(rv)
    if a<1e-12:return np.array([1.,0,0,0])
    u=rv/a; s=np.sin(a/2)
    return np.r_[np.cos(a/2),u*s]
def gyroq(g):
    q=np.array([1.,0,0,0])
    for w in g:q=norm(qm(q,rvq(w*DT)))
    return q
def qrv(q):
    q=norm(q)
    if q[0]<0:q=-q
    a=2*np.arccos(np.clip(q[0],-1,1)); s=np.sqrt(max(1-q[0]**2,0))
    return 2*q[1:] if s<1e-10 else q[1:]/s*a

print("="*72); print("MAVeriCK ML2 - STEP 12: ROTATION AXIS COMPARISON"); print("="*72)
files=sorted(glob.glob(os.path.join(ROOT,"**","S-*.csv"),recursive=True))
print(f"Smartphone files: {len(files)}"); print("Sampling rate: 10 Hz"); print("Window: 1 second = 10 samples\n")
rows=[]
for f in files:
    pair=os.path.splitext(os.path.basename(f))[0][2:]
    try:d=pd.read_csv(f,encoding="utf-8",low_memory=False)
    except UnicodeDecodeError:d=pd.read_csv(f,encoding="latin1",low_memory=False)
    g=d.iloc[:,[x-1 for x in GYRO]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    o=d.iloc[:,[x-1 for x in ORI]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    v=np.all(np.isfinite(g),1)&np.all(np.isfinite(o),1); g,o=g[v],o[v]; n=(len(g)//W)*W
    for i,(gw,ow) in enumerate(zip(g[:n].reshape(-1,W,3),o[:n].reshape(-1,W,3))):
        rg=qrv(gyroq(gw)); ro=qrv(qm(qi(e2q(*ow[0])),e2q(*ow[-1])))
        mg,mo=np.linalg.norm(rg),np.linalg.norm(ro)
        ad=np.nan if mg<=np.deg2rad(1) or mo<=np.deg2rad(1) else np.rad2deg(np.arccos(np.clip(abs(np.dot(rg/mg,ro/mo)),-1,1)))
        rows.append(dict(pair=pair,window_index=i,gyro_rx_deg=np.rad2deg(rg[0]),gyro_ry_deg=np.rad2deg(rg[1]),gyro_rz_deg=np.rad2deg(rg[2]),gyro_rotation_deg=np.rad2deg(mg),orientation_rx_deg=np.rad2deg(ro[0]),orientation_ry_deg=np.rad2deg(ro[1]),orientation_rz_deg=np.rad2deg(ro[2]),orientation_rotation_deg=np.rad2deg(mo),axis_difference_deg=ad))
out=pd.DataFrame(rows)
print("1. ROTATION VECTOR COMPONENT STATISTICS\n"+"-"*72)
for pre in ["gyro","orientation"]:
    print(pre)
    for c in ["rx","ry","rz"]:
        x=out[f"{pre}_{c}_deg"].to_numpy()
        print(f"  {c}: median={np.median(x):.4f}°, p95_abs={np.percentile(abs(x),95):.4f}°, max_abs={max(abs(x)):.4f}°")
a=out.axis_difference_deg.dropna().to_numpy()
print("\n2. AXIS AGREEMENT\n"+"-"*72); print(f"Valid axis comparisons: {len(a)}")
for t in [10,20,30,45,60,90]:
    print(f"Axis difference <= {t:2d}°: {np.sum(a<=t):7d} ({100*np.mean(a<=t):.4f}%)")
print("\n3. LARGE ORIENTATION / SMALL GYRO CASES\n"+"-"*72)
s=out[(out.orientation_rotation_deg>=90)&(out.gyro_rotation_deg<=10)]
print(f"Orientation >=90° AND gyro <=10°: {len(s)} windows")
print(s.sort_values("orientation_rotation_deg",ascending=False).head(20).to_string(index=False))
print("\n4. LARGEST AXIS DISAGREEMENTS\n"+"-"*72)
print(out.dropna(subset=["axis_difference_deg"]).sort_values("axis_difference_deg",ascending=False).head(20).to_string(index=False))
out.to_csv(OUT,index=False)
print("\n"+"="*72); print("STEP 12 COMPLETE"); print("="*72); print(f"Analysis saved: {OUT}"); print("No source, window, or target files were modified.")
