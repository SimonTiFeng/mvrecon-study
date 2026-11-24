"""Apply the >10 m landmark refusal across all scenes and report the net effect."""
import numpy as np, json, pathlib, glob
S=[];R=[];SG=[];RT=[];RA=[];SID=[]
for f in ['work/loo_courtyard_facade.npz','work/loo_delivery_area_electro.npz','work/loo_office_playground_terrace.npz']:
    z=np.load(f,allow_pickle=True)
    S+=list(z['s_lm']);R+=list(z['raw']);SG+=list(z['s_gt']);RT+=list(z['scenes'])
    RA+=list(z['ratios']); SID+=list(z['sids'])
S,R,SG=np.array(S,float),np.array(R,float),np.array(SG,float)
med={}
for f in glob.glob("work/m_*.json")+glob.glob("work/M_*.json"):
    for x in json.loads(pathlib.Path(f).read_text()):
        if x.get("ok",True): med[x["sid"]]=x.get("med_z",999)
C=0.952
def score(i): return float(np.mean(np.abs((S[i]/C)*RA[i]-1.0)))
allc=[score(i) for i in range(len(S))]
keep=[i for i in range(len(S)) if med.get(SID[i],999)<=10]
print("=== Effect of refusing to correct when landmarks exceed 10 m ===")
print(f"{'variant':<24}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>9}")
rw_all=[R[i] for i in range(len(S))]
w_all=sum(1 for i in range(len(S)) if allc[i]<R[i])
print(f"{'no refusal (all)':<24}{len(S):>4}{np.median([R[i] for i in range(len(S))]):>9.4f}"
      f"{np.median(allc):>9.4f}{w_all:>6}/{len(S)}")
rw_k=[R[i] for i in keep]; ca_k=[score(i) for i in keep]
w_k=sum(1 for i in keep if score(i)<R[i])
print(f"{'refuse >10 m':<24}{len(keep):>4}{np.median(rw_k):>9.4f}{np.median(ca_k):>9.4f}{w_k:>6}/{len(keep)}")
print()
print(f"dropped {len(S)-len(keep)} frames, of which "
      f"{sum(1 for i in range(len(S)) if med.get(SID[i],999)>10 and allc[i]>=R[i])} were improving")
